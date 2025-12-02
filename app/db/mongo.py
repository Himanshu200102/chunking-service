# app/db/mongo.py
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import OperationFailure


MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "dataroom")

# Lazy initialization - don't connect at import time
_client = None
_db = None
_indexes_created = False

def get_client():
    """Get or create the MongoDB client."""
    global _client
    if _client is None:
        # Check if we're in testing mode
        if os.getenv("TESTING") == "1":
            import mongomock
            _client = mongomock.MongoClient()
        else:
            _client = MongoClient(MONGO_URI)
    return _client

def get_db():
    """Get or create the MongoDB database."""
    global _db
    if _db is None:
        _db = get_client()[MONGO_DB]
    return _db

# Export for backward compatibility
@property
def client():
    return get_client()

@property  
def db():
    return get_db()

# Make db available as module attribute (for backwards compatibility)
class _LazyDB:
    def __getattr__(self, name):
        return getattr(get_db(), name)

db = _LazyDB()

def ensure_indexes():
    """Create all necessary indexes. Safe to call multiple times."""
    global _indexes_created
    if _indexes_created:
        return
    
    _db = get_db()
    
    # ---- Original indexes ----
    _db.files.create_index([("project_id", ASCENDING), ("deleted_at", ASCENDING)])
    _db.file_versions.create_index([("file_id", ASCENDING), ("version", DESCENDING)])
    _db.file_versions.create_index([("file_id", ASCENDING), ("version", ASCENDING)], unique=True)
    _db.chunks.create_index([("project_id", ASCENDING), ("file_version_id", ASCENDING), ("is_active", ASCENDING)])
    _db.projects.create_index([("members.user_id", ASCENDING)])
    _db.audit.create_index([("project_id", ASCENDING), ("ts", DESCENDING)])
    _db.audit.create_index([("action", ASCENDING), ("ts", DESCENDING)])
    _db.audit.create_index([("file_id", ASCENDING), ("ts", DESCENDING)])
    _db.projects.create_index([("owner_id", ASCENDING)])
    _db.users.create_index([("created_at", ASCENDING)])
    
    # jobs: idempotency + lookups
    _db.jobs.create_index([("version_id", ASCENDING), ("task", ASCENDING)], unique=True, name="uniq_version_task")
    _db.jobs.create_index([("status", ASCENDING), ("enqueue_at", ASCENDING)], name="status_enqueue")
    _db.jobs.create_index([("created_at", DESCENDING)], name="jobs_created_desc")
    _db.chunks.create_index([("file_version_id", 1), ("is_active", 1)])
    _db.chunks.create_index([("chunk_ref", 1)])

    # ---- Targeted additions (fast paths) ----
    # 1) Replace-by-filename: find active file by (project_id, filename)
    _db.files.create_index(
        [("project_id", ASCENDING), ("filename", ASCENDING)],
        name="files_by_project_filename_active",
        partialFilterExpression={"deleted_at": None},
    )
    # 2) Bulk deletes by version_id are common
    _db.chunks.create_index([("file_version_id", ASCENDING)], name="chunks_by_version")
    # 3) Project-wide operations on versions
    _db.file_versions.create_index([("project_id", ASCENDING)], name="versions_by_project")

    # (Optional) faster active-file counts for the 20-file cap
    _db.files.create_index(
        [("project_id", ASCENDING)],
        name="files_active_by_project",
        partialFilterExpression={"deleted_at": None},
    )

    # ---- Chunk indexes ----
    # Query chunks by file_version_id and active status
    _db.chunks.create_index(
        [("file_version_id", ASCENDING), ("is_active", ASCENDING), ("chunk_index", ASCENDING)],
        name="chunks_by_version_active"
    )
    
    # Query chunks by file_id (across all versions)
    _db.chunks.create_index(
        [("file_id", ASCENDING), ("is_active", ASCENDING)],
        name="chunks_by_file_active"
    )
    
    # Add these indexes for the new schema
    _db.chunks.create_index(
        [("dataroom_id", ASCENDING), ("is_active", ASCENDING)],
        name="chunks_by_dataroom"
    )

    _db.chunks.create_index(
        [("doc_id", ASCENDING), ("is_active", ASCENDING), ("chunk_index", ASCENDING)],
        name="chunks_by_doc"
    )

    _db.chunks.create_index(
        [("dataroom_id", ASCENDING), ("section_path", ASCENDING)],
        name="chunks_by_section"
    )

    # Text search - MongoDB only allows ONE text index per collection
    # Drop old text indexes if they exist
    try:
        _db.chunks.drop_index("chunks_text_search")
    except OperationFailure:
        pass  # Index doesn't exist, that's fine
    
    try:
        _db.chunks.drop_index("text_text")  # Default name if created without explicit name
    except OperationFailure:
        pass  # Index doesn't exist, that's fine
    
    # Create compound text index on both text and normalized_text
    # This allows searching both fields with a single index
    _db.chunks.create_index(
        [("text", "text"), ("normalized_text", "text")],
        name="chunks_text_search",
        weights={"normalized_text": 2, "text": 1}  # Prioritize normalized_text
    )
    
    _indexes_created = True

# Only create indexes automatically in production (not at import time during tests)
if os.getenv("TESTING") != "1":
    ensure_indexes()