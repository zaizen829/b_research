"""
識別
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys
import os, glob


classes = ['jui', 'ocha', 'piza', 'pop', 'pre', 'toppo']
num_classes = len(classes)
IMAGE_SIZE = 120

# load model
model = load_model('./six_j.h5')
files = glob.glob('./piza_100/*.jpg')

jui = 0
ocha = 0
piza = 0
pop = 0
pre = 0
toppo = 0

for index, photos in enumerate(files):

    # convert data by specifying file from terminal
    #image = Image.open(sys.argv[1])
    image = Image.open(photos)
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    X = X.astype("float32")


    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    #print(photos[11:])
    print('')
    print(classes[predicted], percentage)
    print('')

    if predicted == 0:
        jui+=1

    if predicted == 1:
        ocha+=1

    if predicted == 2:
        piza+=1

    if predicted == 3:
        pop+=1

    if predicted == 4:
        pre+=1

    if predicted == 5:
        toppo+=1

print('jui:',jui)
print('ocha:',ocha)
print('piza:',piza)
print('pop:',pop)
print('pre:',pre)
print('toppo:',toppo)
