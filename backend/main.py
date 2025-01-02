from contextlib import contextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from dsm2_reader import get_all_data_from_dsm2_dss
from hecdss import HecDss
import tempfile
import pandas as pd
import os


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello world"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(required=True)):
    binary_content = await file.read()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".dss")
    try:
        temp.write(binary_content)
    except Exception as e:
        return {"error": "unable to write temp file"}
    tempfile_name = temp.name
    try:
        dss = HecDss(tempfile_name)
        data = get_all_data_from_dsm2_dss(dss, concat=True)
        cat = dss.get_catalog()
        os.unlink(temp.name)
        return {
            "success": True,
            "filename": file.filename,
            "message": f"dss file processed a total of {len(cat.items)} parts",
        }
    except Exception as e:
        print(f"Error processing file: {str(e)}")  # This will help with debugging
        return {"error": f"unable to process data: {str(e)}"}
