from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from analyzer import load_data, analyze_query
from fastapi import File, UploadFile
import shutil

app = FastAPI()

# CORS (VERY IMPORTANT for Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = load_data()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    return analyze_query(req.question, df)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Reload global dataframe
    global df
    df = load_data()

    return {"message": "File uploaded successfully"}