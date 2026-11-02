"""In-process job queue. One worker: onnxruntime already parallelises inside a single inference,
so running photos one at a time keeps CPU/memory predictable. Swap for Celery/RQ if you need
multiple machines."""
import logging
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from . import faces, store
from .db import Photo, SessionLocal

log = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-worker")


def enqueue(photo_id: str) -> None:
    _pool.submit(_process, photo_id)


def requeue_unfinished() -> None:
    """Photos left pending/processing by a crash or restart get picked up again."""
    with SessionLocal() as s:
        ids = s.scalars(select(Photo.id).where(Photo.status.in_(["pending", "processing"]))).all()
    for photo_id in ids:
        enqueue(photo_id)


def _process(photo_id: str) -> None:
    with SessionLocal() as s:
        photo = s.get(Photo, photo_id)
        if photo is None:
            return
        photo.status = "processing"
        s.commit()
        try:
            found = faces.extract_faces(photo.path)
            store.add_faces(photo.id, photo.event_id, [f.embedding for f in found])
            photo.face_count = len(found)
            photo.status = "done"
        except Exception as exc:  # one bad image must not kill the worker
            log.exception("failed processing %s", photo_id)
            photo.status = "failed"
            photo.error = str(exc)[:500]
        s.commit()
