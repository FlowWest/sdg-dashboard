from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello world"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(required=True)):
    print("upload file --------------------------")
    return {"filename": file.filename}
