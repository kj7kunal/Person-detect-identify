from __future__ import division
import time

import torch 
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import argparse

from torchDarknet import DarknetTorch, nmsYOLO

def parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v","--video", dest='video', help="Input Video for person detection")
	# ap.add_argument("-i","--img", dest='imgpath', help="Input Image Folder for person detection")
	# ap.add_argument("-o","--out", dest='outpath', help="Output Folder for person detection")
	ap.add_argument("-tc","--confthres", dest='confthres', help = "Object Confidence to filter predictions", default=0.65)
	ap.add_argument("-iou","--nmsthres", dest = "nmsthres", help = "NMS IOU Threshhold", default = 0.4)
	ap.add_argument("-c","--cfg", dest='cfg',help="Yolov3 Darknet Config File", default="./cfg/yolov3.cfg")
	ap.add_argument("-w","--weights", dest='weights',help="Yolov3 Pretrained Weights File", default="./yolov3.weights")
	ap.add_argument("-r","--res", dest='res', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)
	return vars(ap.parse_args())

def preprocess(orig_img, dim):
	orig_dim = orig_img.shape[1],orig_img.shape[0]

	#const AR resize with padding
	w, h = dim
	ow, oh = orig_dim
	ratio = min(w/ow,h/oh)
	wnew,hnew = int(ow*ratio),int(oh*ratio)
	img_resized = cv2.resize(orig_img,(wnew,hnew),interpolation=cv2.INTER_CUBIC)
	img = np.full((h,w,3),128)
	img[(h-hnew)//2:hnew+(h-hnew)//2,(w-wnew)//2:wnew+(w-wnew)//2,  :] = img_resized

	#make image RGB and transpose for pytorch BGR -> RGB | H W C -> C H W 
	img_ = img[:,:,::-1].transpose((2,0,1)).copy()
	img_ = torch.from_numpy(img_).float().div(255.).unsqueeze(0)

	return(img_,orig_img,orig_dim)

def mark_classes(x,orig_img):
	label = class_names[int(x[-1])]
	if label=="person":
		cv2.rectangle(orig_img,tuple(x[1:3].int()),tuple(x[3:5].int()),(0,255,0))
		font_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]  
		txtcorner = tuple(x[1:3].int())
		txtcorner = txtcorner[0]+font_sz[0]+2,txtcorner[1]+font_sz[1]+2,
		cv2.rectangle(orig_img,tuple(x[1:3].int()),txtcorner,(255,0,0))
		cv2.putText(orig_img,label,(x[1].int(),txtcorner[1]), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

if __name__=="__main__":
	args = parser()
	confthres = float(args["confthres"])
	nmsthres = float(args["nmsthres"])

	CUDA = torch.cuda.is_available()
	device = torch.device("cuda" if CUDA else "cpu")

	num_classes = 80
	fclasses = open("./data/coco.names","r")
	class_names = fclasses.read().split("\n")[:-1]

	print("Loading network...")
	YOLO = DarknetTorch(args["cfg"])
	YOLO.loadWeights(args["weights"])
	print("Network load success!")

	# Change image resolution from default 320 here
	# Increased resolution means better accuracy.
	# Decreased resolution means faster evaluation.
	YOLO.netparams["height"] = str(args["res"])
	dim = int(YOLO.netparams["height"])
	assert dim % 32 == 0 and dim > 32

	if CUDA:
		print("Running on GPU.")	
	else: 
		print("No GPU found. Running on CPU")
	YOLO = YOLO.to(device)

	# Specify testing 
	YOLO.eval()

	if args["video"]:
		cap = cv2.VideoCapture(args["video"])
		assert cap.isOpened(), 'Cannot open source'

		currFrame=0
		start = time.time()
		while cap.isOpened():
			_,frame = cap.read()
			img,orig_img,orig_dim = preprocess(frame,(dim,dim))
			
			orig_dim = torch.FloatTensor(orig_dim).to(device)
			img = img.to(device)

			with torch.no_grad():
				output = YOLO(Variable(img), CUDA)

			output = nmsYOLO(output, device, confthres, num_classes, nmsthres)	#####################

			# continue loop if no detections
			if type(output) == int:
				currFrame += 1
				print("FPS of the video is %f."%( currFrame / (time.time() - start)))
				cv2.imshow("frame", orig_img)
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
				    break
				continue

			# ratio = torch.min(dim/orig_dim)[0]	########## TORCH 0.5 WARNING

			# #scale output
			# output[:,[1,3]] = (output[:,[1,3]] - (dim - ratio*orig_dim[0])/2)/ratio
			# output[:,[2,4]] = (output[:,[1,3]] - (dim - ratio*orig_dim[1])/2)/ratio

			# for i in range(output.shape[0]):
			# 	output[i,[1,3]] = torch.clamp(output[i,[1,3]], 0.0, orig_dim[0])
			# 	output[i,[2,4]] = torch.clamp(output[i,[2,4]], 0.0, orig_dim[1])

			orig_dim = orig_dim.repeat(output.size(0), 1)
			scaling_factor = torch.min(dim/orig_dim,1)[0].view(-1,1)

			output[:,[1,3]] -= (dim - scaling_factor*orig_dim[:,0].view(-1,1))/2
			output[:,[2,4]] -= (dim - scaling_factor*orig_dim[:,1].view(-1,1))/2

			output[:,1:5] /= scaling_factor

			for i in range(output.shape[0]):
				output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, orig_dim[i,0])
				output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, orig_dim[i,1])


			list(map(lambda x: mark_classes(x,orig_img),output))

			#show video frame
			cv2.imshow("frame", orig_img)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				break
			currFrame += 1
			print("FPS of the video is %f."%( currFrame / (time.time() - start)))


