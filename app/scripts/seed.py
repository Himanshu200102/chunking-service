# scripts/seed.py
import os
from datetime import datetime, timezone
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB  = os.getenv("MONGO_DB", "dataroom")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

now = datetime.now(timezone.utc).isoformat()

project_id = "p_demo"
file_id = "f_demo"
version_id = "v_demo"

# Project: don't overwrite created_at on reseed
db.projects.update_one(
    {"_id": project_id},
    {
        "$set": {
            "name": "Demo Project",
            "owner_id": "u_demo",
            "members": [{"user_id": "u_demo", "role": "owner"}],
            "settings": {"versioning_policy": "supersede"},
            "updated_at": now
        },
        "$setOnInsert": {"created_at": now}
    },
    upsert=True
)

# File
db.files.update_one(
    {"_id": file_id},
    {
        "$set": {
            "project_id": project_id,
            "filename": "demo.pdf",
            "mime": "application/pdf",
            "size": 0,
            "checksum": None,
            "deleted_at": None,
            "updated_at": now
        },
        "$setOnInsert": {"created_at": now}
    },
    upsert=True
)

# File version
db.file_versions.update_one(
    {"_id": version_id},
    {
        "$set": {
            "file_id": file_id,
            "project_id": project_id,
            "version": 1,
            "status": "queued",
            "storage": {
                "raw_uri": None,
                "docling_json_uri": None,
                "text_uri": None,
                "chunks_uri": None
            },
            "error": None,
            "updated_at": now
        },
        "$setOnInsert": {"created_at": now}
    },
    upsert=True
)

print("Seeded: project p_demo, file f_demo, version v_demo")
