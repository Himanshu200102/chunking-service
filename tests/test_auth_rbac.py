
def test_requires_auth_on_project_route(client, seeded_project):
    """Unauthenticated requests should be rejected."""
    r = client.get(f"/projects/{seeded_project}")
    # Accept either 401 (no auth) or 403 (exists but forbidden)
    assert r.status_code in (401, 403), f"Expected 401 or 403, got {r.status_code}"


def test_member_can_read_project(client, seeded_project, auth_header):
    """Project members with reader+ role can access project."""
    r = client.get(f"/projects/{seeded_project}", headers=auth_header)
    assert r.status_code == 200
    assert r.json()["_id"] == seeded_project


def test_non_member_forbidden(client, seeded_project, other_auth_header):
    """Non-members should get 403 Forbidden."""
    r = client.get(f"/projects/{seeded_project}", headers=other_auth_header)
    assert r.status_code == 403