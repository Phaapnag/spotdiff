from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from PIL import Image
import cv2
import numpy as np
import moviepy.editor as mpy
import os
import time
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"message": "Spot Difference API is running"}

@app.post("/upload")
async def upload_images(
    base_image: UploadFile = File(...),
    variant_image:
