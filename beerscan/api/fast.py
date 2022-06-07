from fastapi import FastAPI, File, Request
from beerscan.model.pipeline import main_pipe
import cv2
import numpy as np
import  base64

app = FastAPI()

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.post("/size")
def check_size(file: bytes = File(...)):
    #Check type of image as it arrives.
    print(
         f'''
         --------------------------------------------------------------\n
         --------------------------------------------------------------\n
         \n\n\nAfter being passed to API, file has type: {type(file)}\n
         The file has length {len(file)}
         '''
         )
    # convert to bytes with bytearray, and to np array
    decoded_file=base64.decodebytes(file)

    return f'This file is {len(decoded_file)/1000} Kbytes and type {type(decoded_file)}'


@app.post("/identify_beers")
async def identify_beers(request: Request):

    request_body = await request.json()

    # Decode image
    image = base64.b64decode(request_body['image'])
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, flags=1)

    return main_pipe(image)
