import os
import io
import shutil
import types
import pytest
import mongomock
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import jwt as pyjwt

# === Configure env BEFORE any imports ===
os.environ["TESTING"] = "1"
os.environ.setdefault("MONGO_DB", "dataroom_test")
os.environ.setdefault("JWT_SECRET", "dev-secret")
os.environ.setdefault("LANCEDB_URI", "/tmp/test_lancedb")
os.environ.setdefault("OPENSEARCH_URL", "http://dummy:9200")

@pytest.fixture(scope="session")
def app_instance():
    """Import app after environment is configured."""
    from app.main import app
    return app

@pytest.fixture(autouse=True)
def patch_db_and_clients(monkeypatch, tmp_path):
    """Patch all external dependencies with test doubles."""
    
    # 1) Ensure MongoDB uses mongomock
    import app.db.mongo as mongo_mod
    
    # Reset the module's global state for each test
    mongo_mod._client = None
    mongo_mod._db = None
    mongo_mod._indexes_created = False
    
    # Create fresh mongomock instance for this test
    mock_client = mongomock.MongoClient()
    mock_db = mock_client[os.getenv("MONGO_DB", "dataroom_test")]
    
    # Patch the getters to return our mock
    monkeypatch.setattr(mongo_mod, "get_client", lambda: mock_client)
    monkeypatch.setattr(mongo_mod, "get_db", lambda: mock_db)
    
    # Ensure indexes are created on the mock DB
    mongo_mod.ensure_indexes()

    # 2) Patch deps.get_opensearch to a fake client
    class _FakeOS:
        def delete_by_query(self, index, body, refresh=True, conflicts="proceed"):
            return {"deleted": 123}

    import app.deps as deps
    monkeypatch.setattr(deps, "get_opensearch", lambda: _FakeOS())

    # 3) Patch LanceDB client to a fake in-memory API
    class _FakeTable:
        def __init__(self, name):
            self.name = name
        
        def delete(self, where: str):
            return None

    class _FakeLance:
        def __init__(self):
            self._tables = {"health": _FakeTable("health")}
        
        def table_names(self):
            return list(self._tables.keys())
        
        def open_table(self, name):
            self._tables.setdefault(name, _FakeTable(name))
            return self._tables[name]
        
        def create_table(self, name, data):
            self._tables[name] = _FakeTable(name)

    import app.lancedb_client as ldb
    fake_ldb = _FakeLance()
    monkeypatch.setattr(ldb, "get_lancedb", lambda: fake_ldb)

    # 4) Redirect uploads to tmp so file ops don't touch real FS
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    os.environ["UPLOADS_DIR"] = str(uploads_root)

    yield  # tests run

    # Cleanup
    shutil.rmtree(uploads_root, ignore_errors=True)

@pytest.fixture
def client(app_instance):
    """Test client for making HTTP requests."""
    return TestClient(app_instance)

# === Auth helpers ===
def _token_for(sub="u_demo"):
    """Generate a JWT token for testing."""
    return pyjwt.encode(
        {"sub": sub}, 
        os.getenv("JWT_SECRET", "dev-secret"), 
        algorithm="HS256"
    )

@pytest.fixture
def auth_header():
    """Authorization header for the demo user."""
    return {"Authorization": f"Bearer {_token_for('u_demo')}"}

@pytest.fixture
def other_auth_header():
    """Authorization header for a different user."""
    return {"Authorization": f"Bearer {_token_for('u_other')}"}

# === Seed a demo project ===
@pytest.fixture
def seeded_project():
    """Create a demo project for testing."""
    from app.db.mongo import get_db
    db = get_db()
    
    now = datetime.now(timezone.utc).isoformat()
    db.projects.insert_one({
        "_id": "p_demo",
        "name": "Demo",
        "owner_id": "u_demo",
        "members": [{"user_id": "u_demo", "role": "owner"}],
        "settings": {},
        "created_at": now,
        "updated_at": now
    })
    return "p_demo"