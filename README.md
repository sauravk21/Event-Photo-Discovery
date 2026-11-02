# Event Photo Finder

After an event there are usually hundreds of photos and you only care about the handful you're in. This app finds those for you. Someone uploads the event's photos once, and then anyone can upload a selfie and get back just the photos they appear in.

There's a FastAPI backend, InsightFace for the faces, Qdrant to search them, SQLite for bookkeeping, and one plain HTML page for the UI.

## How it works

When you upload photos, a background worker runs each one through InsightFace.It finds the faces and turns every face into a list  of 512 numbers. Those go into Qdrant, tagged with the photo and the event they came from. When someone uploads a selfie, the same model turns their face into numbers, and Qdrant hands back the closest faces from that event. SQLite just keeps track of which photos are still processing.

Searches only look inside the event you name, so one event's photos never show up in another's.

## Running it

The easiest way is Docker:

```bash
docker compose up -d --build
```

Then open http://localhost:8000. The first start downloads the face model (about 280 MB), so give it a few minutes. Photos, the database and the model live in Docker volumes. `docker compose down` keeps them and `docker compose down -v` wipes them.

If you'd rather run the app on your machine, Qdrant still needs Docker:

```bash
docker compose up -d qdrant
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

I used Python 3.12. insightface compiles a small C++ extension when it installs, so you'll need a compiler (Xcode command line tools on a Mac, build-essential on Linux). The model gets cached in `~/.insightface`.

Pick one way or the other. Both use port 8000, and they keep their photos in different places.

## Using it

Open the page and type an event name, like `birthday-2025`. Lowercase letters, numbers and dashes only. Upload the event's photos and wait for the processing message to finish, then upload a selfie and you'll get your photos back.

The page remembers your event name, and you can also put it in the URL (`/?event=birthday-2025`) to send someone a link that's already set up.

When running locally, the photos end up in `backend/data/photos/<event>/`. Under Docker they're in a volume, not a normal folder on your machine.

## API

FastAPI generates docs at http://localhost:8000/docs. The three endpoints:

- `POST /api/events/{event}/photos` takes one or more images as `files`. It returns straight away and the faces get processed in the background.
- `GET /api/events/{event}` tells you how many photos are done, failed or still processing.
- `POST /api/events/{event}/search` takes a selfie as `selfie` and returns the matching photos, best first.

## Tweaking

- `MATCH_THRESHOLD` (default 0.4) decides how similar two faces need to be. If you're getting the wrong people, raise it towards 0.5. If it misses you in group shots, try 0.35. I haven't tuned it on a real event, so treat 0.4 as a starting point.
- `FACE_MODEL` (default `buffalo_l`) picks the InsightFace model. `buffalo_sc` is a much smaller download if you just want to try things out. Each model gets its own index in Qdrant, so if you switch you'll need to upload the photos again.
- `MAX_IMAGE_SIDE` in `backend/app/config.py` is how far photos get shrunk before face detection. A bigger number finds smaller faces but is slower.