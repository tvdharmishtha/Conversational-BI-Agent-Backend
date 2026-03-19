from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import pandas as pd
from analyzer import analyze_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/current.xlsx"
df = None  # global dataframe

class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global df

    os.makedirs("data", exist_ok=True)

    with open(DATA_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_excel(DATA_PATH)

    return {"message": "File uploaded successfully"}


@app.post("/ask")
def ask(req: QueryRequest):
    global df

    if df is None:
        return {"error": "No file uploaded"}

    return analyze_query(req.question, df)