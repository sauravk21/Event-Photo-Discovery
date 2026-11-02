from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from . import config

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[str] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(index=True)
    original_name: Mapped[str]
    file_name: Mapped[str]  # "<id><ext>", kept in the event's own folder (see path)
    status: Mapped[str] = mapped_column(default="pending")  # pending | processing | done | failed
    face_count: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # event_id is validated as [a-z0-9-] at the API, so it is safe to use as a directory name.
    @property
    def path(self) -> Path:
        return config.UPLOAD_DIR / self.event_id / self.file_name

    @property
    def url(self) -> str:
        return f"/files/{self.event_id}/{self.file_name}"


def init_db() -> None:
    Base.metadata.create_all(engine)
