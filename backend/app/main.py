import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from . import config, faces, jobs, store
from .db import Photo, SessionLocal, init_db

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
EventId = Annotated[str, PathParam(pattern=r"^[a-z0-9-]{1,64}$")]


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    store.ensure_collection()
    faces.warm_up()  # load the model now so the first request isn't the slow one
    jobs.requeue_unfinished()
    yield


app = FastAPI(title="Event Photo Discovery", lifespan=lifespan)


@app.post("/api/events/{event_id}/photos", status_code=202)
def upload_photos(event_id: EventId, files: list[UploadFile]):
    """Save the files, then return immediately; face extraction happens in the background."""
    accepted, rejected = [], []
    with SessionLocal() as s:
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in config.ALLOWED_EXTENSIONS:
                rejected.append(f.filename)
                continue
            photo_id = uuid.uuid4().hex
            photo = Photo(id=photo_id, event_id=event_id, original_name=f.filename, file_name=f"{photo_id}{ext}")
            photo.path.parent.mkdir(parents=True, exist_ok=True)
            with open(photo.path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            s.add(photo)
            accepted.append(photo_id)
        s.commit()
    for photo_id in accepted:
        jobs.enqueue(photo_id)
    return {"accepted": accepted, "rejected": rejected}


@app.get("/api/events/{event_id}")
def event_status(event_id: EventId):
    with SessionLocal() as s:
        rows = s.execute(
            select(Photo.status, func.count(), func.coalesce(func.sum(Photo.face_count), 0))
            .where(Photo.event_id == event_id)
            .group_by(Photo.status)
        ).all()
    counts = {status: (n, nfaces) for status, n, nfaces in rows}
    n = lambda status: counts.get(status, (0, 0))[0]
    return {
        "event_id": event_id,
        "total": sum(c for c, _ in counts.values()),
        "done": n("done"),
        "failed": n("failed"),
        "in_progress": n("pending") + n("processing"),
        "faces": sum(f for _, f in counts.values()),
    }


@app.post("/api/events/{event_id}/search")
def find_my_photos(event_id: EventId, selfie: UploadFile):
    try:
        found = faces.extract_faces(selfie.file)
    except Exception:
        raise HTTPException(422, "Could not read that image.")
    if not found:
        raise HTTPException(422, "No face detected. Try a clearer, front-facing photo.")
    me = max(found, key=lambda f: f.area)  # if others are in frame, assume the biggest face is the user

    hits = store.search(me.embedding, event_id)
    with SessionLocal() as s:
        photos = {p.id: p for p in s.scalars(select(Photo).where(Photo.id.in_([pid for pid, _ in hits])))}
        in_progress = s.scalar(
            select(func.count()).where(Photo.event_id == event_id, Photo.status.in_(["pending", "processing"]))
        )
    return {
        "matches": [
            {"photo_id": pid, "url": photos[pid].url, "score": round(score, 3)}
            for pid, score in hits
            if pid in photos
        ],
        "photos_still_processing": in_progress,  # results may be incomplete while this is > 0
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # StaticFiles requires the directory at mount time
app.mount("/files", StaticFiles(directory=config.UPLOAD_DIR), name="files")
