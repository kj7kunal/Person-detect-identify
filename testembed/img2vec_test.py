import sys
import os
import time

from img2vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2

import csv

input_path = './'
models=[('resnet18',512,0.81),('resnet34',512,0.83),('resnet50',2048,0.84),('resnet101',2048,0.85),('resnet152',2048,0.86)]
# models=[('resnet34',512,0.84),('resnet50',2048,0.84)]

def findSimilarity(mod='resnet34',path=input_path,ol = 512,thres=0.8):
    csvfile = path+mod+'.csv'
    start = time.time()
    img2vec = Img2Vec(model = mod,layer_output_size=ol)
    pics = {}
    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            # img = cv2.imread(os.path.join(input_path, filename))
            img = Image.open(os.path.join(input_path, filename))
            vec = img2vec.get_vec(img) 
            pics[filename] = vec
    nettime = (time.time()-start)/len(pics)
    print("Average network time per pic: ",nettime)

    res = []
    rowhead = sorted(list(pics.keys()))
    res.append(["%.1fms-%d%%"%(nettime*1000,thres*100)]+rowhead)
    for i in rowhead:
        # print(i)
        sims = [i]
        # start = time.time()
        for j in rowhead:
            if i==j:
                sims.append(-1)
                continue
            sim = cosine_similarity(pics[i].reshape((1, -1)), pics[j].reshape((1, -1)))[0][0]
            sims.append(sim if sim>thres else 0)
        # print("Average time for similarity with other images per image: ",i,(time.time()-start)/len(sims))
        res.append(sims)

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(res)

for mod,olayer_size,thres in models:
    print(mod,olayer_size,thres)
    findSimilarity(mod,input_path,olayer_size,thres)
