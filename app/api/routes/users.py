# app/routes/users.py
from fastapi import APIRouter, Depends
from app.auth.userctx import current_user
from app.db.mongo import db

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/me")
def me(u=Depends(current_user)):
    doc = db.users.find_one({"_id": u["user_id"]}, {"_id": 1, "name": 1, "email": 1, "created_at": 1})
    return {"user_id": u["user_id"], **(doc or {})}
