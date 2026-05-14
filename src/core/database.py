"""
SleepAI Database Schema and Connection
"""
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Optional

from .config import DATABASE_URL
from .time_utils import utc_now

Base = declarative_base()


class ConceptModel(Base):
    """SQLAlchemy model for concepts"""
    __tablename__ = "concepts"

    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # JSON string of vector
    novelty = Column(Float, default=0.5)
    emotional = Column(Float, default=0.0)
    task_relevance = Column(Float, default=0.5)
    repetition = Column(Float, default=0.5)
    importance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now)
    last_accessed = Column(DateTime, default=utc_now)
    access_count = Column(Integer, default=0)
    strength = Column(Float, default=1.0)
    state = Column(String, default="active")  # active, consolidating, archived, suppressed
    version_root = Column(String, nullable=True)
    version_parent = Column(String, nullable=True)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    is_current_version = Column(Boolean, default=True)
    context_tags = Column(Text, nullable=True)


class RelationModel(Base):
    """SQLAlchemy model for relations"""
    __tablename__ = "relations"

    id = Column(String, primary_key=True)
    subject_id = Column(String, nullable=False)
    predicate = Column(String, nullable=False)
    object_id = Column(String, nullable=False)
    strength = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utc_now)
    bidirectional = Column(Boolean, default=False)


class EpisodeModel(Base):
    """SQLAlchemy model for episodes (working memory)"""
    __tablename__ = "episodes"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=utc_now)
    concept_ids = Column(Text, nullable=True)  # JSON array of concept IDs
    raw_content = Column(Text, nullable=False)
    context = Column(Text, nullable=True)  # JSON dict
    importance_json = Column(Text, nullable=True)
    state = Column(String, default="active")
    source = Column(String, default="user")


class SleepCycleModel(Base):
    """SQLAlchemy model for sleep cycles"""
    __tablename__ = "sleep_cycles"

    id = Column(String, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    nrem_duration = Column(Float, default=0.0)
    rem_duration = Column(Float, default=0.0)
    memories_consolidated = Column(Integer, default=0)
    memories_forgotten = Column(Integer, default=0)
    dreams_json = Column(Text, nullable=True)  # JSON array


def init_db():
    """Initialize database connection and create tables"""
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal


def get_session():
    """Get a database session"""
    engine, SessionLocal = init_db()
    return SessionLocal()
