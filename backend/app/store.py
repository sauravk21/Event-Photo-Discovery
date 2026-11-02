import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from . import config

client = QdrantClient(url=config.QDRANT_URL)


def ensure_collection() -> None:
    if client.collection_exists(config.COLLECTION):
        return
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
    )
    # Every search is scoped to one event, so index the field we filter on.
    client.create_payload_index(config.COLLECTION, "event_id", models.PayloadSchemaType.KEYWORD)


def add_faces(photo_id: str, event_id: str, embeddings: list[np.ndarray]) -> None:
    # Deterministic ids make re-processing a photo an overwrite instead of a duplicate.
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{photo_id}/{i}")),
            vector=emb.tolist(),
            payload={"photo_id": photo_id, "event_id": event_id},
        )
        for i, emb in enumerate(embeddings)
    ]
    if points:
        client.upsert(config.COLLECTION, points)


def search(embedding: np.ndarray, event_id: str, limit: int = 500) -> list[tuple[str, float]]:
    """Return (photo_id, best_score) pairs, best match first, one entry per photo."""
    hits = client.query_points(
        config.COLLECTION,
        query=embedding.tolist(),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="event_id", match=models.MatchValue(value=event_id))]
        ),
        score_threshold=config.MATCH_THRESHOLD,
        limit=limit,
    ).points

    best: dict[str, float] = {}
    for h in hits:  # a photo with two similar faces would otherwise appear twice
        pid = h.payload["photo_id"]
        best[pid] = max(best.get(pid, 0.0), h.score)
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)
