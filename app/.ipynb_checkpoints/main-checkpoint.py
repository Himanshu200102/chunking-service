from fastapi import FastAPI
from app.api.routes import dataroom

app = FastAPI(title="DataRoom FastAPI")

# Register routers
app.include_router(dataroom.router, prefix="/api", tags=["DataRoom"])

@app.get("/")
def root():
    return {"message": "Welcome to DataRoom API"}
