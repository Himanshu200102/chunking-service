import os
from typing import List
from fastapi import UploadFile

UPLOAD_DIR = "uploads"


async def save_new_collection(folder_name: str, files: List[UploadFile]):
    folder_path = os.path.join(UPLOAD_DIR, folder_name.replace(" ", "_"))
    os.makedirs(folder_path, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)
    return uploaded_files


def list_documents_in_collection(folder_name: str):
    folder_path = os.path.join(UPLOAD_DIR, folder_name.replace(" ", "_"))
    if not os.path.exists(folder_path):
        return []
    return os.listdir(folder_path)


async def add_to_collection(folder_name: str, files: List[UploadFile]):
    folder_path = os.path.join(UPLOAD_DIR, folder_name.replace(" ", "_"))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    uploaded_files = []
    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)
    return uploaded_files


def list_all_collections():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    return os.listdir(UPLOAD_DIR)


async def update_document_in_collection(folder_name: str, file_name: str, new_file: UploadFile):
    """Replace an existing document in the collection."""
    return False


def delete_document_from_collection(folder_name: str, file_name: str):
    """Delete a specific document from a collection."""
    return False


def delete_collection(folder_name: str):
    """Delete the entire folder (DataRoom)."""
    return False