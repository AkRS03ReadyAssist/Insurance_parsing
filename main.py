from fastapi import FastAPI, UploadFile, File, HTTPException, status
from PyPDF2 import PdfReader
import io
import json
from langchain_text_splitters import CharacterTextSplitter
import re
from output_generation import extract_details
app = FastAPI()
text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200,length_function=len)
files = {}  # Dictionary to store uploaded PDFs
comsumer_files={}
@app.post('/pdf', status_code=status.HTTP_202_ACCEPTED)
async def pdf_submit(file: UploadFile = File()):
    file_name = file.filename
    try:
        # Read file content and process it
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text

        files[file_name] = text  # Store extracted text in the dictionary
        text_final=text_splitter.split_text(text)
        return extract_details(text=text_final)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.get('/pdf')
def pdf_get():
    return files  # Return stored filenames
@app.get('/pdf/{file_name}')
def pdf_get(file_name:str):
    if file_name not in files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail='file not found')
    text_final=text_splitter.split_text(files[file_name])
    return extract_details(text=text_final)