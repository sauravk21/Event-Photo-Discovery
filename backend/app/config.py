import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
UPLOAD_DIR = DATA_DIR / "photos"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_DIM = 512  # InsightFace ArcFace embedding size

FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")
# Embeddings from different models are all 512-d but not comparable. Keying the collection by model
# means switching FACE_MODEL gives an empty index (re-upload) rather than silently wrong matches.
COLLECTION = f"faces_{FACE_MODEL}"
MIN_DET_SCORE = 0.5  # drop low-confidence detections (blurry background "faces")
MAX_IMAGE_SIDE = 1600  # downscale before detection; faster and barely affects accuracy

# Cosine similarity cut-off for "same person". ArcFace: ~0.35 is permissive, ~0.5 is strict.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.4"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
