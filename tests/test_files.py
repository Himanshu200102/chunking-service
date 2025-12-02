import io
import os

def _fake_pdf_bytes():
    return b"%PDF-1.4\n%Fake\n1 0 obj\n<<>>\nendobj\n%%EOF"

def test_upload_creates_file_and_version(client, auth_header, seeded_project, monkeypatch, tmp_path):
    # Patch storage to write into tmp_path and avoid absolute /app/uploads
    from app.api.routes import files as files_mod

    def fake_ensure_dir(path): os.makedirs(path, exist_ok=True)
    def fake_save_upload_file(file, dest_path):
        # write to tmp and return path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        data = file.file.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        file.file.seek(0)
        return (len(data), dest_path)
    def fake_sha256(_fp): return "sha256:deadbeef"

    monkeypatch.setattr(files_mod, "ensure_dir", fake_ensure_dir, raising=True)
    monkeypatch.setattr(files_mod, "save_upload_file", fake_save_upload_file, raising=True)
    monkeypatch.setattr(files_mod, "sha256_of_file", lambda f: "sha256:deadbeef", raising=True)

    # Upload
    files = {"file": ("demo.pdf", io.BytesIO(_fake_pdf_bytes()), "application/pdf")}
    r = client.post(f"/projects/{seeded_project}/files", headers=auth_header, files=files)
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["file_id"].startswith("f_")
    assert j["version"] == 1
    assert j["checksum"] == "sha256:deadbeef"

def test_replace_creates_new_version_same_file_id(client, auth_header, seeded_project, monkeypatch):
    # Same monkeypatch as above for storage
    from app.api.routes import files as files_mod
    monkeypatch.setattr(files_mod, "ensure_dir", lambda p: os.makedirs(p, exist_ok=True), raising=True)
    monkeypatch.setattr(files_mod, "save_upload_file",
                        lambda file, dest: (len(file.file.read()), dest), raising=True)
    monkeypatch.setattr(files_mod, "sha256_of_file", lambda f: "sha256:beadfeed", raising=True)

    files = {"file": ("demo.pdf", io.BytesIO(b"v1"), "application/pdf")}
    r1 = client.post(f"/projects/{seeded_project}/files", headers=auth_header, files=files)
    assert r1.status_code == 201, r1.text
    fid = r1.json()["file_id"]

    files2 = {"file": ("demo.pdf", io.BytesIO(b"v2"), "application/pdf")}
    r2 = client.post(f"/projects/{seeded_project}/files?replace=true", headers=auth_header, files=files2)
    assert r2.status_code == 201, r2.text
    j2 = r2.json()
    assert j2["file_id"] == fid
    assert j2["version"] == 2

def test_file_cap_20_blocks_new_file_ids(client, auth_header, seeded_project, monkeypatch):
    from app.db.mongo import db
    # Seed 20 active files
    for i in range(20):
        db.files.insert_one({"_id": f"f_{i}", "project_id": seeded_project, "filename": f"a{i}.pdf",
                             "deleted_at": None})
    from app.api.routes import files as files_mod
    monkeypatch.setattr(files_mod, "ensure_dir", lambda p: os.makedirs(p, exist_ok=True), raising=True)
    monkeypatch.setattr(files_mod, "save_upload_file",
                        lambda file, dest: (len(file.file.read()), dest), raising=True)
    monkeypatch.setattr(files_mod, "sha256_of_file", lambda f: "sha256:xx", raising=True)

    files = {"file": ("new.pdf", io.BytesIO(b"v"), "application/pdf")}
    r = client.post(f"/projects/{seeded_project}/files", headers=auth_header, files=files)
    assert r.status_code == 409
    assert "file_limit_exceeded" in r.text

def test_hard_delete_file_removes_docs_and_audits(client, auth_header, seeded_project, monkeypatch):
    from app.api.routes import files as files_mod
    monkeypatch.setattr(files_mod, "ensure_dir", lambda p: os.makedirs(p, exist_ok=True), raising=True)
    monkeypatch.setattr(files_mod, "save_upload_file",
                        lambda file, dest: (len(file.file.read()), dest), raising=True)
    monkeypatch.setattr(files_mod, "sha256_of_file", lambda f: "sha256:yy", raising=True)

    import io
    files = {"file": ("demo.pdf", io.BytesIO(b"data"), "application/pdf")}
    r = client.post(f"/projects/{seeded_project}/files", headers=auth_header, files=files)
    fid = r.json()["file_id"]

    # Delete the file
    d = client.delete(f"/projects/{seeded_project}/files/{fid}", headers=auth_header)
    assert d.status_code == 200, d.text
    # Verify deletions & audit
    from app.db.mongo import db
    assert db.files.count_documents({"_id": fid}) == 0
    assert db.file_versions.count_documents({"file_id": fid}) == 0
    assert db.audit.count_documents({"file_id": fid, "action": "file_deleted"}) == 1
