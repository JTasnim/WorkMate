from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from src.backend.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    All models inherit from this — SQLAlchemy uses it to track table definitions.
    """
    pass


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite with FastAPI
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def create_tables() -> None:
    """Create all tables defined in models. Safe to call multiple times — skips existing tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a database session.
    Closes the session automatically after the request finishes.
    Used as: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()