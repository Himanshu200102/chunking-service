# app/auth/routes.py
import os, jwt
from fastapi import APIRouter, HTTPException, status, Query
from .jwt import JWT_ALG  # reuse your constants if you have them

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")

router = APIRouter(prefix="/auth", tags=["auth"])
# app/auth/routes.py
import uuid
from fastapi import APIRouter, HTTPException, status, Query
from app.db.mongo import db
from app.auth.jwt import mint_token
from datetime import datetime, timezone

router = APIRouter(prefix="/auth", tags=["auth"])

def _now(): return datetime.now(timezone.utc).isoformat()
def _new_user_id() -> str: return f"u_{uuid.uuid4().hex[:10]}"

@router.post("/signup", status_code=201)
def signup(name: str | None = Query(None), email: str | None = Query(None)):
    """
    Create a user with a random UUID-based id and return a JWT.
    No password; keep for dev/testing.
    """
    uid = _new_user_id()
    db.users.insert_one({"_id": uid, "name": name, "email": email, "created_at": _now(), "updated_at": _now()})
    return {"user_id": uid, "access_token": mint_token(uid), "token_type": "bearer"}

@router.post("/login", status_code=200)
def login(user_id: str = Query(..., description="Your user_id returned by /auth/signup")):
    """
    Issue a fresh JWT for an existing user_id.
    """
    if not db.users.find_one({"_id": user_id}):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"user_id": user_id, "access_token": mint_token(user_id), "token_type": "bearer"}

@router.get("/dev-token")
def dev_token(sub: str = Query(..., description="User id (e.g., u_demo)")):
    """
    DEV ONLY. Issues a JWT for testing. In prod, replace with real IdP/login.
    """
    if not sub:
        raise HTTPException(status_code=400, detail="missing sub")
    token = jwt.encode({"sub": sub}, JWT_SECRET, algorithm=JWT_ALG if 'JWT_ALG' in globals() else "HS256")
    return {"access_token": token, "token_type": "bearer"}
