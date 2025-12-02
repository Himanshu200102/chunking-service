import os
from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from app.services.dataroom_service import (
    save_new_collection,
    list_documents_in_collection,
    add_to_collection,
    list_all_collections
)

router = APIRouter()

UPLOAD_BASE_PATH = "uploads"

@router.post("/upload_folder")
async def upload_folder(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Create a new DataRoom (folder) and upload multiple documents."""
    result = await save_new_collection(folder_name, files)
    return {"message": f"Created collection '{folder_name}'", "files_uploaded": result}


@router.get("/list_all_documents")
async def list_all_documents(folder_name: str):
    """List all documents in a collection."""
    result = list_documents_in_collection(folder_name)
    return {"folder_name": folder_name, "documents": result}


@router.post("/add_to_my_collection")
async def add_to_my_collection(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Add more files to an existing collection."""
    result = await add_to_collection(folder_name, files)
    return {"message": f"Added files to '{folder_name}'", "files_uploaded": result}


@router.get("/list_all_collections")
async def list_all_collections_api():
    """List all existing DataRoom collections."""
    result = list_all_collections()
    return {"collections": result}


@router.put("/update_document")
async def update_document(
    folder_name: str = Form(...),
    file_name: str = Form(...),
    new_file: UploadFile = File(...)
):
    """Replace an existing document with a new one."""
    result = await update_document_in_collection(folder_name, file_name, new_file)
    if not result:
        raise HTTPException(status_code=404, detail="File or folder not found")
    return {"message": f"Updated '{file_name}' in '{folder_name}'"}


@router.delete("/delete_document")
async def delete_document(folder_name: str, file_name: str):
    """Delete a single document from a collection."""
    result = delete_document_from_collection(folder_name, file_name)
    if not result:
        raise HTTPException(status_code=404, detail="File or folder not found")
    return {"message": f"Deleted '{file_name}' from '{folder_name}'"}


@router.delete("/delete_collection")
async def delete_collection_api(folder_name: str):
    """Delete an entire DataRoom collection (folder)."""
    result = delete_collection(folder_name)
    if not result:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"message": f"Deleted entire collection '{folder_name}'"}