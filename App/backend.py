from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os
from pydantic import BaseModel
from doc_embedding import process_documents
from query_vectorDB import query_question

app = FastAPI()

# CORS
# Required as streamlit frontend (port 8501) is different from the FastAPI backend (port 8000).
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploading Documents
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Write files inside the temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
            
            # Call the process documents function
            result = process_documents(temp_dir)
            
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Querying data
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        print("Trying quer_question")
        response = query_question(request.question)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
