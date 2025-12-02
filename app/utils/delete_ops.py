import os, shutil
from typing import Iterable, Tuple

def rm_rf(path: str) -> None:
    """Recursively delete a path if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def delete_upload_tree(base_dir: str, project_id: str, file_id: str) -> str:
    """
    Deletes /uploads/projects/<pid>/raw/<file_id>/ (all versions).
    Returns the deleted root path for logging.
    """
    root = os.path.join(base_dir, "projects", project_id, "raw", file_id)
    rm_rf(root)
    return root
