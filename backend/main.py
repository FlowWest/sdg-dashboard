from contextlib import contextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from dsm2_reader import get_all_data_from_dsm2_dss
from hecdss import HecDss
from io import BytesIO
import tempfile
import pandas as pd
import os


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello world"}


#
# @contextmanager
# def temporary_file(content: bytes):
#     """Context manager for handling temporary files"""
#     temp = tempfile.NamedTemporaryFile(delete=False)
#     try:
#         temp.write(content)
#         temp.close()
#         yield temp.name  # This is where the 'with' block executes
#     finally:
#         os.unlink(temp.name)  # This cleanup happens no matter what


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(required=True)):
    print(f"the file received {file}")
    binary_content = await file.read()
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp.write(binary_content)
    except Exception as e:
        return {"error": "unable to write temp file"}
    tempfile_name = temp.name
    try:
        dss = HecDss(tempfile_name)
        data = get_all_data_from_dsm2_dss(dss)
        print(f"the len of the data {len(data)}")
        cat = dss.get_catalog()

        os.unlink(temp.name)
        return {
            "success": True,
            "filename": file.filename,
            "message": f"the len of data was {cat.items}",
        }
    except Exception as e:
        print(f"Error processing file: {str(e)}")  # This will help with debugging
        return {"error": f"unable to process data: {str(e)}"}
