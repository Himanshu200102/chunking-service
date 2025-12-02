def test_indexes_exist():
    from app.db.mongo import db
    # Check key ones by name/spec
    files_idx = [i["key"] for i in db.files.list_indexes()]
    fv_idx = [i["key"] for i in db.file_versions.list_indexes()]
    chunks_idx = [i["key"] for i in db.chunks.list_indexes()]
    projects_idx = [i["key"] for i in db.projects.list_indexes()]
    audit_idx = [i["key"] for i in db.audit.list_indexes()]

    # Basic presence checks
    assert {"project_id": 1, "deleted_at": 1} in files_idx
    assert {"file_id": 1, "version": -1} in fv_idx
    assert {"project_id": 1, "file_version_id": 1, "is_active": 1} in chunks_idx
    assert {"members.user_id": 1} in projects_idx
    assert {"project_id": 1, "ts": -1} in audit_idx
