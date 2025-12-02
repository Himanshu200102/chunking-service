# app/auth/userctx.py
from datetime import datetime, timezone
from fastapi import Depends
from app.auth.jwt import bearer, decode_token
from app.db.mongo import db

def _now(): return datetime.now(timezone.utc).isoformat()

async def current_user(auth=Depends(bearer)):
    claims = decode_token(auth)
    sub = claims.get("sub")
    if not sub:
        raise ValueError("Missing 'sub' in token")

    # Ensure user exists (safe if already created)
    db.users.update_one(
        {"_id": sub},
        {"$setOnInsert": {"created_at": _now()}, "$set": {"updated_at": _now()}},
        upsert=True
    )
    return {"user_id": sub, "claims": claims}
