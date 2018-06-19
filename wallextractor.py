
from keras.applications.resnet50 import ResNet50
from keras import Model
from PIL import Image
import os
import keras.applications.imagenet_utils as imutl
import keras.backend.common as backend
import keras.utils as utils
import numpy as np
import itertools as it
def isImgFile(filename : str) -> bool:
    exts = os.path.splitext(filename)[-1].lower()
    return exts in [".jpg",".png",".jpeg"]
def extract(inputPath:str, outputPath:str) :
    for dirpath,_,filenames in os.walk(inputPath):
        outputDir = os.path.join(outputPath, os.path.relpath(dirpath, inputPath))
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)
        for filename in filter(isImgFile, filenames):
            yield os.path.join(dirpath,filename), os.path.join(outputDir, filename)
def genimages(tuplesInputOutput, sizex, sizey):
    for inputFile,_ in tuplesInputOutput:
        try:
            img = Image.open(inputFile)
            yield np.asarray(img.resize(size=(sizex,sizey),resample=Image.BILINEAR).convert("RGB"))# / 255.
        except Exception as e:
            print(e)
            yield np.zeros((sizex,sizey,3))
def group(itr, num):
    return it.takewhile(lambda x: len(x) > 0, map(lambda x: list(x()), it.repeat(lambda: it.islice(itr,num))))

def count_by(pred,itr):
    count = 0
    for i in itr:
        if pred(i):
            count += 1
    return count

import math
import random
import shutil
if __name__ == "__main__":
    filepaths = list(extract("C:\WallClassification\downloads","C:\WallClassification\walls"))
    random.shuffle(filepaths)
    batch_size = 64
    batchGenerator = map(lambda imgs: np.asarray(imgs), group(genimages(filepaths,224,224),batch_size))
    resnet = ResNet50(input_shape=(224,224,3))
    predict = resnet.predict_generator(batchGenerator, 
        steps=math.ceil(len(filepaths)/batch_size), 
        # steps = 20,
        workers=4,
        verbose=1)
    predict = np.argmax(predict,axis=1)
    print(predict, len(predict), count_by(lambda x: x == 825,predict))
    loop = map(lambda y: y[1], filter(lambda x: x[0] == 825, zip(predict,filepaths)))
    for input,output in loop:
        shutil.copyfile(input,output,)
    