# SleepAI Phase 1: Core Memory System
## Complete Technical Documentation

**Version**: 1.0
**Date**: April 2026
**Status**: Production Ready
**Project**: SleepAI
**Code Name**: Core Memory Engine

> Historical note (April 29, 2026): this document is a Phase 1 snapshot. Current system status is tracked in `docs/PROJECT_STATUS.md` and includes Phase 1 through Phase 6 hardening work.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Data Models](#4-data-models)
5. [API Reference](#5-api-reference)
6. [Database Schema](#6-database-schema)
7. [Configuration](#7-configuration)
8. [Testing](#8-testing)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Security Considerations](#10-security-considerations)
11. [Known Limitations](#11-known-limitations)
12. [Implementation Notes](#12-implementation-notes)
13. [Future Roadmap](#13-future-roadmap)
14. [Glossary](#14-glossary)

---

## 1. Executive Summary

### 1.1 Project Overview

**SleepAI** is a brain-inspired memory system for AI assistants that mimics how human memory works: attention-based encoding, value-based importance tagging, working memory, long-term storage, and (in later phases) sleep consolidation.

**Phase 1 Focus**: Core memory infrastructure without sleep consolidation.

### 1.2 What Phase 1 Does

```
INPUT: "My name is Saish, I prefer morning meetings"
           ↓
ATTENTION FILTER: What matters?
           ↓
MEANING ENCODER: Extract concepts (person, preference)
           ↓
VALUE TAGGER: Tag importance (novelty, emotional, task-relevance)
           ↓
WORKING MEMORY: Fast, limited (7 items)
           ↓
LONG-TERM MEMORY: Persistent SQLite storage
           ↓
RETRIEVAL: Query by text, ranked by importance
```

### 1.3 Key Achievements (Phase 1)

| Metric | Value |
|--------|-------|
| Concepts extracted | 1789+ stored |
| Encoder latency | 0.01ms per extraction |
| Search latency | 0.36ms per query |
| Memory capacity | 7 items (enforced) |
| Concurrent writes | 10 threads safe |
| Sessions persisted | 100% across restarts |

### 1.4 Differentiation from Other Systems

| System | SleepAI Phase 1 Advantage |
|--------|---------------------------|
| **ChatGPT** | Remembers across sessions |
| **MemGPT** | No explicit sleep needed, faster retrieval |
| **Stateless retrieval libraries** | Simpler architecture, lighter dependencies |
| **RAG** | No vector database required, simpler stack |
| **Simple vector DB** | Meaning-based extraction, not just embedding similarity |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           SLEEPAI                                        │
│                    Brain-Inspired Memory System                          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────────┐    ┌───────────────────────┐    ┌───────────────────┐
│   INPUT LAYER     │    │   PROCESSING LAYER   │    │   STORAGE LAYER   │
│                   │    │                      │    │                   │
│ • Raw text input  │───▶│ • MeaningEncoder     │───▶│ • WorkingMemory   │
│ • User context    │    │ • ValueTagger        │    │ • SQLite (LTM)    │
│ • Session info    │    │ • Attention Filter   │    │ • Graph structure │
└───────────────────┘    └───────────────────────┘    └───────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   RETRIEVAL LAYER    │
                        │                      │
                        │ • Query parser       │
                        │ • Ranked results     │
                        │ • Importance filter  │
                        └───────────────────────┘
```

### 2.2 Component Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ MeaningEncoder  │ ── Extracts semantic concepts from text
│ (LLM-powered)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ValueTagger   │ ── Multi-dimensional importance scoring
│ (4 dimensions)  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│Working │ │LongTerm │
│Memory  │ │Memory   │
│(fast)  │ │(SQLite) │
└────────┘ └──────────┘
```

### 2.3 Data Flow During Store Operation

```python
1. user_input = "My name is Saish"
2. concepts = encoder.extract(user_input)
   # → [Concept(type="person", description="Person: Saish"),
   #    Concept(type="preference", description="Preference: name Saish")]
3. for concept in concepts:
       concept.importance = value_tagger.tag(concept)
   # → ImportanceVector(novelty=0.9, emotional=0.0, ...)
4. episode = Episode(concept_ids=[c.id for c in concepts], raw_content=user_input)
5. working_memory.store(episode)
6. for concept in concepts:
       db.save_concept(concept)
```

### 2.4 Data Flow During Retrieve Operation

```python
1. query = "What's my name?"
2. results = db.search_concepts("name", limit=5)
3. ranked = sorted(results, key=lambda x: x['importance'], reverse=True)
4. return ranked[:limit]
```

---

## 3. Component Specifications

### 3.1 MeaningEncoder

**Purpose**: Convert raw text into semantic concept graph.

**Location**: `src/core/encoder.py`

**Class**: `MeaningEncoder`

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `extract(text)` | `str` | `List[Concept]` | Extract concepts from text |
| `extract_with_relations(text)` | `str` | `Tuple[List[Concept], List[Relation]]` | Extract concepts AND relations |
| `compute_similarity(text1, text2)` | `str, str` | `float` | Cosine similarity between texts |

**Extraction Strategy**:
1. Pattern-based heuristics (no LLM required)
2. Regex patterns for names, preferences, facts
3. Sentence transformer embeddings for similarity

**Example**:
```python
encoder = MeaningEncoder()
text = "My name is Saish, I prefer morning meetings"
concepts = encoder.extract(text)
# Output:
# [Concept(type="person", description="Person: Saish"),
#  Concept(type="preference", description="Preference: morning meetings")]
```

**Supported Patterns**:
- Name: `r"(?:my name is|i am|i'm|I'm)\s+([A-Z][a-z]+)"`
- Preference: `r"(?:prefer|like|hate|dislike|enjoy)\s+([^.!?]+)"`
- Age: `r"(\d+)\s+(?:years?\s+old|yo)"`
- Work: `r"(?:work on|working on|building|developing)\s+([A-Z][a-zA-Z]+)"`

### 3.2 ValueTagger

**Purpose**: Assign multi-dimensional importance to concepts.

**Location**: `src/core/value_tagger.py`

**Class**: `ValueTagger`

**Dimensions**:

| Dimension | Range | Description |
|-----------|-------|-------------|
| `novelty` | 0.0-1.0 | How new is this? (0=seen many times, 1=completely new) |
| `emotional` | -1.0 to 1.0 | Positive (1.0) to negative (-1.0) |
| `task_relevance` | 0.0-1.0 | Important for current task/goals |
| `repetition` | 0.0-1.0 | How many times reinforced (0=first time, 1=many times) |
| `overall` | 0.0-1.0 | Weighted average (computed property) |

**Overall Score Formula**:
```
overall = (novelty × 0.3) + ((emotional + 1) / 2 × 0.2) + (task_relevance × 0.3) + (repetition × 0.2)
```

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `tag(concept, context)` | `Concept, Dict` | `ImportanceVector` | Tag single concept |
| `tag_batch(concepts, context)` | `List[Concept], Dict` | `List[ImportanceVector]` | Tag multiple concepts |
| `update_history(concept)` | `Concept` | `None` | Record seen concept |

**Example**:
```python
tagger = ValueTagger()
concept = Concept(type="person", description="Person: Saish")
importance = tagger.tag(concept, context={'task': 'assistant'})
# Output: ImportanceVector(novelty=0.9, emotional=0.0, task_relevance=0.6, repetition=0.0)
```

### 3.3 WorkingMemory

**Purpose**: Fast-access, limited-capacity short-term memory (hippocampal equivalent).

**Location**: `src/core/working_memory.py`

**Class**: `WorkingMemory`

**Capacity**: 7 items (Miller's Law - human short-term memory limit)

**Data Structure**: `collections.deque` (FIFO with maxlen)

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `store(episode)` | `Episode` | `None` | Store episode, evict oldest if full |
| `retrieve(query, limit)` | `str, int` | `List[Episode]` | Get relevant episodes |
| `get_recent(n)` | `int` | `List[Episode]` | Get n most recent |
| `is_full()` | None | `bool` | Check if at capacity |
| `size()` | None | `int` | Current count |
| `remove_episode(id)` | `str` | `bool` | Remove specific episode |
| `clear()` | None | `None` | Clear all episodes |
| `to_dict()` | None | `Dict` | Serialize for storage |
| `from_dict(data)` | `Dict` | `WorkingMemory` | Deserialize |

**Example**:
```python
wm = WorkingMemory(capacity=7)
episode = Episode(concept_ids=["c1", "c2"], raw_content="My name is Saish")
wm.store(episode)
print(wm.size())  # 1
print(wm.is_full())  # False
```

### 3.4 LongTermMemory (Concept Graph)

**Purpose**: Persistent semantic graph storage (cortical equivalent).

**Location**: `src/core/long_term_memory.py`

**Class**: `LongTermMemory`

**Storage**: NetworkX directed graph + SQLite

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_concept(concept)` | `Concept` | `Concept` | Add concept to graph |
| `add_relation(relation)` | `Relation` | `Relation` | Add relation between concepts |
| `get_concept(id)` | `str` | `Concept` | Retrieve by ID |
| `get_related_concepts(id, depth)` | `str, int` | `List[Concept]` | Get connected concepts |
| `search_by_text(query, limit)` | `str, int` | `List[Concept]` | Text search |
| `search_by_embedding(query, limit)` | `List[float], int` | `List[Concept]` | Vector similarity search |
| `update_concept(id, updates)` | `str, Dict` | `bool` | Update fields |
| `update_importance(id, importance)` | `str, ImportanceVector` | `bool` | Update importance |
| `remove_concept(id, soft)` | `str, bool` | `bool` | Delete concept |
| `get_stats()` | None | `Dict` | Memory statistics |

**Graph Structure**:
```python
self.graph = nx.DiGraph()  # Directed graph
# Nodes: concept IDs
# Edges: relations (subject → object) with predicate as edge attribute
```

### 3.5 SQLiteMemory

**Purpose**: SQLite-based persistent storage for concepts and episodes.

**Location**: `src/core/sqlite_db.py`

**Class**: `SQLiteMemory`

**Database File**: `data/sleepai.db`

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `save_concept(concept)` | `Concept` | `bool` | Store concept |
| `get_concept(id)` | `str` | `Dict` | Retrieve concept |
| `search_concepts(query, limit)` | `str, int` | `List[Dict]` | Text search |
| `save_episode(episode)` | `Episode` | `bool` | Store episode |
| `get_recent_episodes(limit)` | `int` | `List[Dict]` | Get recent episodes |
| `get_stats()` | None | `Dict` | Statistics |

---

## 4. Data Models

### 4.1 ConceptType (Enum)

```python
class ConceptType(str, Enum):
    PERSON = "person"        # Named person
    PREFERENCE = "preference" # User preference
    FACT = "fact"          # Factual information
    EVENT = "event"        # Temporal event
    RELATION = "relation"  # Relationship
    OBJECT = "object"      # Physical object
    LOCATION = "location"  # Place
    ABSTRACT = "abstract"  # Abstract concept
```

### 4.2 PredicateType (Enum)

```python
class PredicateType(str, Enum):
    HAS_PROPERTY = "has_property"
    PREFERS = "prefers"
    LOCATED_AT = "located_at"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
```

### 4.3 MemoryState (Enum)

```python
class MemoryState(str, Enum):
    ACTIVE = "active"          # Currently in use
    CONSOLIDATING = "consolidating"  # Being transferred to LTM
    ARCHIVED = "archived"     # Archived for later retrieval
    SUPPRESSED = "suppressed" # Soft deleted (can be recovered)
```

### 4.4 ImportanceVector

```python
class ImportanceVector(BaseModel):
    novelty: float = Field(default=0.5, ge=0, le=1)
    emotional: float = Field(default=0.0, ge=-1, le=1)
    task_relevance: float = Field(default=0.5, ge=0, le=1)
    repetition: float = Field(default=0.5, ge=0, le=1)

    @property
    def overall(self) -> float:
        return (
            self.novelty * 0.3 +
            (self.emotional + 1) / 2 * 0.2 +
            self.task_relevance * 0.3 +
            self.repetition * 0.2
        )
```

### 4.5 Concept

```python
class Concept(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ConceptType
    description: str
    embedding: Optional[List[float]] = None
    importance: ImportanceVector = Field(default_factory=ImportanceVector)
    state: MemoryState = Field(default=MemoryState.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    strength: float = 1.0
```

### 4.6 Episode

```python
class Episode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    concept_ids: List[str] = []
    raw_content: str
    context: Dict[str, Any] = {}
    importance: ImportanceVector = Field(default_factory=ImportanceVector)
    state: MemoryState = MemoryState.ACTIVE
    source: str = "user"  # user, assistant, system
```

### 4.7 Relation

```python
class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str
    predicate: PredicateType
    object_id: str
    strength: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    bidirectional: bool = False
```

---

## 5. API Reference

### 5.1 FastAPI Endpoints

**Base URL**: `http://localhost:8000`

#### POST `/memory/`
Store new information in memory.

**Request**:
```json
{
  "text": "My name is Saish",
  "context": {"task": "personal_assistant"},
  "source": "user"
}
```

**Response**:
```json
{
  "success": true,
  "concepts_created": 2,
  "episode_id": "3a3756f8-...",
  "message": "Stored 2 concepts from: My name is Saish..."
}
```

#### GET `/memory/?query=NAME&limit=5`
Query memory for relevant information.

**Request**: `GET /memory/?query=Saish&limit=5`

**Response**:
```json
{
  "results": [
    {
      "type": "concept",
      "id": "...",
      "description": "Person: Saish",
      "concept_type": "person",
      "importance": 0.72,
      "created_at": "2026-04-20T...",
      "source": "long_term"
    }
  ],
  "memory_stats": {
    "total_concepts": 1789,
    "working_memory_size": 3,
    "suppressed_count": 0
  }
}
```

#### GET `/memory/stats`
Get memory statistics.

**Response**:
```json
{
  "long_term": {
    "total_concepts": 1789,
    "total_relations": 142,
    "suppressed_count": 5
  },
  "working_memory_size": 3,
  "working_memory_capacity": 7
}
```

#### DELETE `/memory/working`
Clear working memory (used during sleep phase).

**Response**:
```json
{
  "success": true,
  "message": "Working memory cleared"
}
```

#### POST `/memory/consolidate`
Manually trigger consolidation from working to long-term memory.

**Response**:
```json
{
  "success": true,
  "consolidated": 6,
  "episodes_processed": 3
}
```

### 5.2 Health Check

#### GET `/`
Root endpoint.

**Response**:
```json
{
  "name": "SleepAI",
  "version": "0.1.0",
  "status": "running",
  "description": "Brain-inspired memory with sleep consolidation"
}
```

#### GET `/health`
Health check.

**Response**:
```json
{
  "status": "healthy"
}
```

---

## 6. Database Schema

### 6.1 SQLite Tables

#### concepts
| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| type | TEXT | NOT NULL | ConceptType value |
| description | TEXT | NOT NULL | Human-readable description |
| embedding | TEXT | NULL | JSON array of floats |
| novelty | REAL | DEFAULT 0.5 | Importance dimension |
| emotional | REAL | DEFAULT 0.0 | Importance dimension |
| task_relevance | REAL | DEFAULT 0.5 | Importance dimension |
| repetition | REAL | DEFAULT 0.5 | Importance dimension |
| importance_score | REAL | DEFAULT 0.0 | Computed overall |
| created_at | TEXT | NOT NULL | ISO timestamp |
| last_accessed | TEXT | NOT NULL | ISO timestamp |
| access_count | INTEGER | DEFAULT 0 | Access counter |
| strength | REAL | DEFAULT 1.0 | Connection strength |
| state | TEXT | DEFAULT 'active' | MemoryState value |

#### relations
| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| subject_id | TEXT | NOT NULL | Source concept ID |
| predicate | TEXT | NOT NULL | PredicateType value |
| object_id | TEXT | NOT NULL | Target concept ID |
| strength | REAL | DEFAULT 1.0 | Relation strength |
| created_at | TEXT | NOT NULL | ISO timestamp |
| bidirectional | INTEGER | DEFAULT 0 | Boolean |

#### episodes
| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| timestamp | TEXT | NOT NULL | ISO timestamp |
| concept_ids | TEXT | NULL | JSON array |
| raw_content | TEXT | NOT NULL | Original text |
| context | TEXT | NULL | JSON dict |
| importance_json | TEXT | NULL | ImportanceVector JSON |
| state | TEXT | DEFAULT 'active' | MemoryState value |
| source | TEXT | DEFAULT 'user' | user/assistant/system |

#### sleep_cycles
| Column | Type | Constraints | Description |
|--------|------|------------|-------------|
| id | TEXT | PRIMARY KEY | UUID |
| start_time | TEXT | NOT NULL | ISO timestamp |
| end_time | TEXT | NULL | ISO timestamp |
| nrem_duration | REAL | DEFAULT 0.0 | Seconds |
| rem_duration | REAL | DEFAULT 0.0 | Seconds |
| memories_consolidated | INTEGER | DEFAULT 0 | Count |
| memories_forgotten | INTEGER | DEFAULT 0 | Count |
| dreams_json | TEXT | NULL | JSON array |

### 6.2 Database File Location

```
SleepAI/
└── data/
    └── sleepai.db  # SQLite database
```

---

## 7. Configuration

### 7.1 Environment Variables

Create `.env` file in project root:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=sleepai

# LLaMA (for future LLM extraction)
LLAMA_MODEL_PATH=/path/to/llama-7b.q4.bin
LLAMA_N_CTX=2048
LLAMA_N_THREADS=4

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 7.2 Config File

**Location**: `src/core/config.py`

**Key Settings**:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKING_MEMORY_CAPACITY` | 7 | Max items in working memory |
| `IMPORTANCE_THRESHOLD` | 0.3 | Below this = forget candidate |
| `NOVELTY_THRESHOLD` | 0.7 | Above this = new concept |
| `EMBEDDING_DIM` | 384 | MiniLM-L6 embedding dimension |
| `SLEEP_ENTROPY_THRESHOLD` | 0.9 | Sleep trigger threshold |
| `SLEEP_CONFLICT_THRESHOLD` | 0.3 | Conflict density threshold |
| `NREM_DOWNSCALE_FACTOR` | 0.8 | NREM consolidation strength |

---

## 8. Testing

### 8.1 Test Files

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_encoder.py` | 5 | MeaningEncoder unit tests |
| `tests/test_value_tagger.py` | 6 | ValueTagger unit tests |
| `tests/test_working_memory.py` | 8 | WorkingMemory unit tests |
| `tests/test_memory_pipeline.py` | - | End-to-end integration test |
| `tests/test_brutal.py` | 26 | Comprehensive stress test |

### 8.2 Running Tests

```bash
# All tests
cd SleepAI
source venv/bin/activate
pytest tests/ -v

# Specific test file
pytest tests/test_brutal.py -v

# Quick smoke test
python tests/test_memory_pipeline.py
```

### 8.3 Test Coverage (Brutal Suite)

| Category | Tests |
|----------|-------|
| Edge cases (empty, long, special chars, unicode, SQL, XSS) | 6 |
| Working memory (capacity, importance, consistency) | 4 |
| Stress tests (rapid, large, concurrent) | 4 |
| Data integrity (persistence, cross-session) | 3 |
| Search quality (partial, case-insensitive, ranking) | 3 |
| Deletion/forgetting (soft, hard, threshold) | 3 |
| Performance benchmarks | 3 |
| **Total** | **26** |

### 8.4 Test Results (Latest)

```
======================================================================
SLEEPAI BRUTAL TEST REPORT
======================================================================
Total tests: 26
Passed: 26
Failed: 0

✅ ALL TESTS PASSED! SleepAI is production ready.
======================================================================
```

---

## 9. Performance Benchmarks

### 9.1 Encoder Performance

| Metric | Value | Conditions |
|--------|-------|------------|
| Latency | 0.01ms | Per extraction |
| Throughput | ~100,000/sec | Sequential |
| Memory | ~200MB | Model loaded |
| Model | sentence-transformers/all-MiniLM-L6-v2 | 384 dimensions |

### 9.2 Search Performance

| Metric | Value | Conditions |
|--------|-------|------------|
| Latency | 0.36ms | Per search |
| Throughput | ~2,700/sec | Sequential |
| Database size | 1789+ concepts | Tested |

### 9.3 Working Memory Operations

| Metric | Value | Conditions |
|--------|-------|------------|
| Store latency | 0.01ms | Per operation |
| Retrieve latency | 0.01ms | Per operation |
| Capacity | 7 items | Enforced |
| Overflow handling | FIFO eviction | Oldest evicted |

### 9.4 Concurrent Write Performance

| Metric | Value | Conditions |
|--------|-------|------------|
| Threads | 10 | ThreadPoolExecutor |
| Operations per thread | 20 | 200 total |
| Time | ~0.5s | Full test |
| Data integrity | 100% | No corruption |

### 9.5 Storage Capacity

| Metric | Value |
|--------|-------|
| Max concepts | Unlimited (SQLite limit) |
| Tested concepts | 1789+ |
| Episode storage | Unlimited |
| Database file growth | ~1MB per 1000 concepts |

---

## 10. Security Considerations

### 10.1 Input Sanitization

| Threat | Mitigation | Status |
|--------|------------|--------|
| SQL Injection | Parameterized queries | ✅ Protected |
| XSS | Text stored as data, not executed | ✅ Protected |
| Path traversal | No file system access | ✅ Protected |
| Unicode exploits | Full Unicode support | ✅ Protected |
| Large input DoS | 10KB+ handled gracefully | ✅ Protected |

### 10.2 SQL Injection Prevention

The system uses SQLite with parameterized queries:

```python
# Safe - no SQL injection possible
cursor.execute("SELECT * FROM concepts WHERE description LIKE ?", (f"%{query}%",))
```

### 10.3 XSS Prevention

All stored content is treated as data, not code. No HTML rendering of user content.

---

## 11. Known Limitations

### 11.1 Phase 1 Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **No sleep consolidation** | Memories don't reorganize | Phase 2 |
| **No forgetting** | Memory grows forever | Manual cleanup |
| **No LLM extraction** | Only heuristic patterns | LLaMA integration in Phase 2 |
| **SQLite only** | Not distributed | PostgreSQL support planned |
| **No authentication** | Anyone can access | API key in future |
| **Single-user only** | No multi-user support | Future multi-tenancy |

### 11.2 Known Edge Cases

| Case | Current Behavior | Desired Behavior |
|------|------------------|-------------------|
| Very long text | Extracts only first concept | Better extraction |
| Ambiguous text | May misclassify | LLM-powered disambiguation |
| Duplicate concepts | Creates duplicates | Better deduplication |
| Non-English text | Limited patterns | Enhanced multilingual |

### 11.3 Not Implemented (Future Phases)

- [ ] Sleep trigger mechanism
- [ ] NREM consolidation
- [ ] REM dreaming
- [ ] Intentional forgetting
- [ ] Multi-session persistence with consolidation
- [ ] LLM-powered extraction
- [ ] PostgreSQL support
- [ ] Authentication
- [ ] Multi-agent memory sync

---

## 12. Implementation Notes

### 12.1 Project Structure

```
SleepAI/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py         # Pydantic data models
│   │   ├── config.py         # Configuration
│   │   ├── database.py       # PostgreSQL models (future)
│   │   ├── sqlite_db.py     # SQLite implementation
│   │   ├── encoder.py      # MeaningEncoder
│   │   ├── value_tagger.py  # ValueTagger
│   │   ├── working_memory.py # WorkingMemory
│   │   └── long_term_memory.py # LongTermMemory
│   ├── sleep/               # (Phase 2)
│   │   ├── __init__.py
│   │   ├── trigger.py       # Sleep trigger
│   │   ├── nrem.py          # NREM consolidation
│   │   └── rem.py           # REM dreaming
│   ├── retrieval/            # (Future)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI app
│   │   └── memory.py       # Memory endpoints
├── data/                     # Database files
│   └── sleepai.db
├── models/                   # ML models (future)
├── tests/
│   ├── __init__.py
│   ├── test_encoder.py
│   ├── test_value_tagger.py
│   ├── test_working_memory.py
│   ├── test_memory_pipeline.py
│   └── test_brutal.py
├── research/                 # Paper analyses & docs
├── requirements.txt
├── README.md
└── .env.example
```

### 12.2 Dependencies

```
# Core
sentence-transformers>=2.2.0  # Embeddings
networkx>=3.0                 # Graph memory
faiss-cpu>=1.7.0             # Vector search (future)
sqlalchemy>=2.0             # ORM
psycopg2-binary>=2.9.0      # PostgreSQL (future)

# API
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0

# Sleep Modules (Phase 2)
torch>=2.0.0

# Testing
pytest>=7.4.0
```

### 12.3 Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_brutal.py

# Start API
python -m src.api.main
```

### 12.4 Usage Example

```python
from src.core.encoder import MeaningEncoder
from src.core.value_tagger import ValueTagger
from src.core.working_memory import WorkingMemory
from src.core.sqlite_db import get_memory

# Initialize
encoder = MeaningEncoder()
value_tagger = ValueTagger()
working_memory = WorkingMemory()
db = get_memory()

# Store
text = "My name is Saish, I prefer morning meetings"
concepts = encoder.extract(text)
for c in concepts:
    c.importance = value_tagger.tag(c)
    db.save_concept(c)

# Retrieve
results = db.search_concepts("Saish", limit=5)
print(results)
```

---

## 13. Future Roadmap

### Phase 2: Sleep Consolidation (Weeks 5-8)

| Component | Description |
|-----------|-------------|
| **Sleep Trigger** | Entropy-based + conflict density detection |
| **NREM Module** | Hebbian consolidation + synaptic downscaling |
| **Forgetting Module** | Value-based forgetting threshold |
| **Sleep Cycle** | Wake → NREM → REM → Wake loop |

### Phase 3: Advanced Sleep (Weeks 9-12)

| Component | Description |
|-----------|-------------|
| **REM Dreaming** | Generative replay for novel combinations |
| **Cross-Session** | Memory persists and consolidates across sessions |
| **Multi-Modal** | Support for images, audio in memory |

### Future (Post-Phase 3)

- [ ] LLM-powered semantic extraction
- [ ] PostgreSQL for distributed deployment
- [ ] Authentication & multi-user
- [ ] Multi-agent memory synchronization
- [ ] Real-time adaptation
- [ ] Mobile app integration

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **Concept** | Semantic unit of memory (person, preference, fact, etc.) |
| **Episode** | Single event/utterance stored in working memory |
| **Working Memory** | Fast, limited-capacity short-term storage (hippocampal equivalent) |
| **Long-Term Memory** | Persistent storage (cortical equivalent) |
| **Importance Vector** | Multi-dimensional scoring (novelty, emotional, task_relevance, repetition) |
| **MeaningEncoder** | Component that extracts semantic concepts from text |
| **ValueTagger** | Component that computes importance of concepts |
| **NREM** | Non-REM sleep stage (consolidation phase) |
| **REM** | REM sleep stage (dreaming/synthesis phase) |
| **Hebbian Plasticity** | "Neurons that fire together wire together" learning rule |
| **Synaptic Downscaling** | Proportional weakening of all synapses during sleep |
| **Forgetting** | Active removal of low-importance memories |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial Phase 1 documentation |

---

*End of Document*
*SleepAI Phase 1 Technical Documentation*
