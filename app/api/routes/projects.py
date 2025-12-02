# app/routes/projects.py
from fastapi import APIRouter, HTTPException, Request, Query, Body
from datetime import datetime, timezone
import uuid
import os
from pydantic import BaseModel

from app.db.mongo import db
from app.utils.stream import stream_error
from app.utils.delete_ops import rm_rf
from app.lancedb_client import get_lancedb
from app.deps import get_opensearch

router = APIRouter(prefix="/projects", tags=["projects"])

# ---------- Request Models ----------
class CreateProjectRequest(BaseModel):
    user_id: str = "temp_user_001"
    name: str

class DeleteProjectRequest(BaseModel):
    user_id: str = "temp_user_001"

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _pid() -> str:
    return f"p_{uuid.uuid4().hex[:10]}"

def _aid() -> str:
    return f"a_{uuid.uuid4().hex[:10]}"

# ---------- LIST MY PROJECTS (useful in Swagger) ----------
@router.get("", status_code=200)
def list_my_projects(
    user_id: str = Query(default="temp_user_001", description="User ID")
):
    cur = db.projects.find({"owner_id": user_id}, {"_id": 1, "name": 1, "created_at": 1})
    return [{"project_id": p["_id"], "name": p["name"], "created_at": p.get("created_at")} for p in cur]

# ---------- GET SINGLE PROJECT ----------
@router.get("/{project_id}", status_code=200)
def get_project(
    project_id: str,
    user_id: str = Query(default="temp_user_001", description="User ID"),
):
    """
    Get a single project by ID. User must be a member of the project.
    
    Returns:
        The project document
        
    Raises:
        403: If user is not a member of the project
        404: If project doesn't exist
    """
    # Find the project
    prj = db.projects.find_one({"_id": project_id})
    if not prj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if user is a member
    is_member = any(m.get("user_id") == user_id for m in prj.get("members", []))
    
    if not is_member:
        raise HTTPException(
            status_code=403, 
            detail="Access denied: you are not a member of this project"
        )
    
    return prj

# ---------- CREATE PROJECT ----------
@router.post("", status_code=201)
def create_project(
    request_body: CreateProjectRequest = Body(...),
):
    user_id = request_body.user_id
    name = request_body.name

    # Enforce: user can own at most 2 projects
    owned_count = db.projects.count_documents({"owner_id": user_id})
    if owned_count >= 2:
        return stream_error([
            "error: project_limit_exceeded",
            f"detail: user '{user_id}' already owns {owned_count} projects (max 2).",
            "action: delete an existing project or transfer ownership."
        ], status_code=409)

    # Create project
    pid = _pid()
    now = _now()
    db.projects.insert_one({
        "_id": pid,
        "name": name,
        "owner_id": user_id,
        "members": [{"user_id": user_id, "role": "owner"}],
        "settings": {"versioning_policy": "supersede"},
        "created_at": now,
        "updated_at": now,
    })
    return {"project_id": pid, "name": name, "owner_id": user_id}

# ---------- HARD DELETE PROJECT (cascade + audit) ----------
@router.delete("/{project_id}", status_code=200)
def hard_delete_project(
    request: Request,
    project_id: str,
    user_id: str = Query(..., description="User ID performing the deletion"),
):
    """
    Hard delete an entire project:
      - delete all file chunks (Mongo), OpenSearch docs, LanceDB rows
      - delete all file_versions, files, and the project doc
      - delete uploaded blobs under /uploads/projects/<pid>
      - write a single audit record
    """
    prj = db.projects.find_one({"_id": project_id})
    if not prj:
        raise HTTPException(status_code=404, detail="Project not found")

    # Collect IDs for cascade
    file_ids = [f["_id"] for f in db.files.find({"project_id": project_id}, {"_id": 1})]
    version_ids = [v["_id"] for v in db.file_versions.find({"project_id": project_id}, {"_id": 1})]

    # Mongo: chunks
    chunks_deleted = int(db.chunks.delete_many({"project_id": project_id}).deleted_count or 0)

    # OpenSearch: delete-by-query (best effort)
    os_deleted = 0
    try:
        if version_ids:
            os_client = get_opensearch()
            resp = os_client.delete_by_query(
                index="chunks",
                body={"query": {"terms": {"file_version_id": version_ids}}},
                refresh=True,
                conflicts="proceed",
            )
            os_deleted = int(resp.get("deleted", 0) or 0)
    except Exception:
        os_deleted = -1

    # LanceDB: best effort delete
    lance_deleted = 0
    try:
        ldb = get_lancedb()
        if "chunks" in ldb.table_names():
            tbl = ldb.open_table("chunks")
            for vid in version_ids:
                tbl.delete(where=f"file_version_id = '{vid}'")
    except Exception:
        lance_deleted = -1

    # Mongo: versions, files, project
    versions_deleted = int(db.file_versions.delete_many({"project_id": project_id}).deleted_count or 0)
    files_deleted    = int(db.files.delete_many({"project_id": project_id}).deleted_count or 0)
    project_deleted  = int(db.projects.delete_one({"_id": project_id}).deleted_count or 0)

    # Blobs on disk
    uploads_root = os.path.join("/app/uploads", "projects", project_id)
    rm_rf(uploads_root)  # ignore errors inside

    # Audit
    actor_ip = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None
    audit_doc = {
        "_id": _aid(),
        "project_id": project_id,
        "action": "project_deleted",
        "actor_user_id": user_id,  # Changed from request_body.user_id
        "file_ids": file_ids,
        "version_ids": version_ids,
        "counts": {
            "chunks": chunks_deleted,
            "versions": versions_deleted,
            "files": files_deleted,
            "os_docs": os_deleted,
            "lance_rows": lance_deleted,
            "project_docs": project_deleted
        },
        "blobs_root_deleted": uploads_root,
        "client": {"ip": actor_ip, "user_agent": user_agent},
        "ts": _now(),
    }
    db.audit.insert_one(audit_doc)

    return {"ok": True, "project_id": project_id, "audit_id": audit_doc["_id"]}