from __future__ import division
import time

import torch 
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import argparse

from torchDarknet import DarknetTorch

def parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v","--video", dest='vpath', help="Input Video for person detection")
	ap.add_argument("-i","--img", dest='imgpath', help="Input Image for person detection")
	ap.add_argument("-tc","--confthres", dest='confthres', help = "Object Confidence to filter predictions", default=0.5)
	ap.add_argument("-iou","--nmsthres", dest = "nmsthres", help = "NMS IOU Threshhold", default = 0.4)
	ap.add_argument("-c","--cfg", dest='cfgFile',help="Yolov3 Darknet Config File", default="./cfg/yolov3.cfg")
	ap.add_argument("-w","--weights", dest='wPath',help="Yolov3 Pretrained Weights File", default="./yolov3.weights")
	ap.add_argument("-r","--res", dest='res', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)
	return vars(ap.parse_args())

def preprocess(orig_img, dim):
	orig_dim = orig_img.shape[1],orig_img.shape[0]

	#const AR resize with padding
	w, h = dim
	ow, oh = orig_dim
	ratio = min(w/ow,h/oh)
	wnew,hnew = int(ow*ratio),int(oh*ratio)
	img_resized = cv2.resize(img,(wnew,hnew),interpolation=cv2.INTER_CUBIC)
	img = np.full((h,w,3),128)
	img[(h-hnew)//2:hnew+(h-hnew)//2,(w-wnew)//2:wnew+(w-wnew)//2,  :] = img_resized

	#make image RGB and transpose for pytorch BGR -> RGB | H W C -> C H W 
	img_ = img[:,:,::-1].transpose((2,0,1)).copy()
	img_ = torch.from_numpy(img_).float().div(255.).unsqueeze(0)

	return(img_,orig_img,orig_dim)




if __name__=="__main__":
	args = parser()
	confthres = float(args["confthres"])
	nmsthres = float(args["nmsthres"])

	CUDA = torch.cuda.is_available()
	device = torch.device("cuda" if CUDA else "cpu")

	print("Loading network...")
	YOLO = DarknetTorch(args["cfg"])
	YOLO.loadWeights(args["weights"])
	print("Network load success!")

	# Change image resolution from default 320 here
	# Increased resolution means better accuracy.
	# Decreased resolution means faster evaluation.
	YOLO.netparams["height"] = str(args["res"])
	dim = int(YOLO.netparams["height"])
	assert dim % 32 == 0 
	assert dim > 32

	if CUDA:
		print("Running on GPU.")	
	else: 
		print("No GPU found. Running on CPU")
	YOLO.to(device)


	if args["video"]:
		cap = cv2.VideoCapture(args["video"])
		assert cap.isOpened(), 'Cannot open source'

		i=0
		start = time.time()
		while cap.isOpened():
			_,frame = cap.read()
			img,orig_img,orig_dim = preprocess(frame,dim)
			
			# orig_dim = torch.FloatTensor(orig_dim).repeat(1,2)
			orig_dim = torch.FloatTensor(orig_dim).to(device)
			img = img.to(device)

			with torch.no_grad():
				output = net(Variable(img), CUDA)
			






	YOLO()