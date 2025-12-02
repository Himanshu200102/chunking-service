import os
from opensearchpy import OpenSearch

# OpenSearch singleton
_opensearch = None

def get_opensearch() -> OpenSearch:
    """Return a singleton OpenSearch client."""
    global _opensearch
    if _opensearch is None:
        # Check if we're in testing mode
        if os.getenv("TESTING") == "1":
            # Return a mock for tests (will be patched by conftest)
            class _MockOS:
                def delete_by_query(self, index, body, refresh=True, conflicts="proceed"):
                    return {"deleted": 0}
            _opensearch = _MockOS()
        else:
            os_url = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
            _opensearch = OpenSearch(os_url)
    return _opensearch