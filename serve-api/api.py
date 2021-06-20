from typing import Optional
import os
from fastapi import FastAPI, File, UploadFile
from mmcv import Config, DictAction
from mmcls.apis import init_model, inference_model
from inference import inference_model
import mmcv

app = FastAPI()

cfg = Config.fromfile(
    'mmclassification/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py')
model = init_model(cfg,'model.pth')
if (os.getenv('MODEL') != None) and (os.getenv('MODEL')=='food101'):
    cfg.model.head.num_classes = 101
    model.CLASSES = []
    with open('classes.txt') as f:
        for x in f.readlines():
            model.CLASSES.append(x[:-1])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(file:bytes = File(...)):
    img = mmcv.imfrombytes(file)
    return inference_model(model,img)