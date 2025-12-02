# app/auth/rbac.py
from fastapi import Depends, HTTPException, Request
from app.db.mongo import db
from app.auth.userctx import current_user  # uses Authorization: Bearer ... or X-User-Id

# Role order for comparison
ORDER = {"reader": 0, "editor": 1, "owner": 2}

def require_project_role(project_id_param: str, min_role: str):
    """
    Dependency factory that ensures the caller has at least `min_role`
    on the project identified by `project_id_param` (path or query).
    Returns a dict: {"user_id", "role", "project_id"} on success.

    - 401: handled by current_user when no/invalid credentials.
    - 400: missing project id param.
    - 404: project not found.
    - 403: not a member or role below requirement.
    """
    if min_role not in ORDER:
        raise ValueError(f"Invalid min_role '{min_role}'. Valid: {list(ORDER.keys())}")

    async def _dep(
        request: Request,
        u = Depends(current_user),  # resolves and upserts user; raises 401 if missing
    ):
        user_id = u["user_id"]

        # Support both path and query param for project id
        project_id = request.path_params.get(project_id_param) or request.query_params.get(project_id_param)
        if not project_id:
            raise HTTPException(status_code=400, detail=f"Missing '{project_id_param}'")

        prj = db.projects.find_one({"_id": project_id}, {"members": 1})
        if not prj:
            raise HTTPException(status_code=404, detail="Project not found")

        role = next((m.get("role") for m in prj.get("members", []) if m.get("user_id") == user_id), None)
        if role is None:
            raise HTTPException(status_code=403, detail="Not a project member")

        if ORDER.get(role, -1) < ORDER[min_role]:
            raise HTTPException(status_code=403, detail=f"Requires role {min_role}+")
        return {"user_id": user_id, "role": role, "project_id": project_id}

    return _dep
