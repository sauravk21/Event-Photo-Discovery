from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps

from . import config


@dataclass
class Face:
    embedding: np.ndarray  # L2-normalised, so cosine similarity == dot product
    area: float  # bbox area in pixels; used to pick the main face in a selfie


@lru_cache(maxsize=1)
def _analyzer() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.FACE_MODEL,
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],  # skip landmarks/gender/age models
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def warm_up() -> None:
    _analyzer()


def _load_bgr(source: Path | BinaryIO) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")  # phones store rotation in EXIF
    img.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE))
    return np.asarray(img)[:, :, ::-1]  # RGB -> BGR, which is what InsightFace expects


def extract_faces(source: Path | BinaryIO) -> list[Face]:
    faces = _analyzer().get(_load_bgr(source))
    return [
        Face(
            embedding=f.normed_embedding.astype(np.float32),
            area=float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        for f in faces
        if f.det_score >= config.MIN_DET_SCORE
    ]
