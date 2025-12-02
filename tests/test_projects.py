def test_create_projects_with_cap(client, auth_header):
    # First
    r1 = client.post("/projects?name=Alpha", headers=auth_header)
    assert r1.status_code == 201, r1.text
    # Second
    r2 = client.post("/projects?name=Beta", headers=auth_header)
    assert r2.status_code == 201, r2.text
    # Third should stream an error 409
    r3 = client.post("/projects?name=Gamma", headers=auth_header)
    assert r3.status_code == 409
    assert "project_limit_exceeded" in r3.text

def test_hard_delete_project_cascades(client, auth_header, seeded_project):
    # Create a file + version in the project
    from app.db.mongo import db
    db.files.insert_one({"_id":"f_x","project_id":seeded_project,"filename":"a.pdf","deleted_at":None})
    db.file_versions.insert_one({"_id":"v_f_x_1","file_id":"f_x","project_id":seeded_project,"version":1})
    db.chunks.insert_one({"_id":"c1","project_id":seeded_project,"file_version_id":"v_f_x_1","is_active":True})
    # Delete project
    r = client.delete(f"/projects/{seeded_project}", headers=auth_header)
    assert r.status_code == 200, r.text
    # Project gone
    assert db.projects.count_documents({"_id": seeded_project}) == 0
    # Files gone
    assert db.files.count_documents({"project_id": seeded_project}) == 0
    # Versions gone
    assert db.file_versions.count_documents({"project_id": seeded_project}) == 0
    # Chunks gone
    assert db.chunks.count_documents({"project_id": seeded_project}) == 0
    # Audit written
    assert db.audit.count_documents({"project_id": seeded_project, "action": "project_deleted"}) == 1
