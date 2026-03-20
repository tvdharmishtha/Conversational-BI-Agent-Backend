from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import pandas as pd
from analyzer import analyze_query, get_column_info, suggest_currency

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


class ColumnsResponse(BaseModel):
    columns: list
    column_types: dict


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global df

    os.makedirs("data", exist_ok=True)

    with open(DATA_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_excel(DATA_PATH)

    # Get column information
    column_info = get_column_info(df)
    
    # Get currency suggestion
    currency_info = suggest_currency(df)

    return {
        "message": "File uploaded successfully",
        "columns": column_info["columns"],
        "column_types": column_info["column_types"],
        "row_count": len(df),
        "currency": currency_info
    }


@app.get("/columns")
def get_columns():
    global df

    if df is None:
        return {"error": "No file uploaded", "columns": [], "column_types": {}}

    column_info = get_column_info(df)

    return {
        "columns": column_info["columns"],
        "column_types": column_info["column_types"],
        "row_count": len(df),
        "preview": df.head(5).to_dict(orient='records')
    }


@app.post("/ask")
def ask(req: QueryRequest):
    global df

    if df is None:
        return {"error": "No file uploaded"}

    return analyze_query(req.question, df)
