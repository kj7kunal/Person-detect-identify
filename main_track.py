from __future__ import division
import time

import torch 
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import argparse

from torchDarknet import DarknetTorch, nmsYOLO
from img2vec import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity

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
	ap.add_argument("--ivmod", dest='ivmod', help = "Model to use for img2vec embeddings", default='resnet50')
	ap.add_argument("-ttr","--trackthres", dest='trackthres', help = "Similarity threshold for Tracking", default=0.84)
	
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

def mark_all_persons(x,orig_img,i):
	label = class_names[int(x[-1])]
	if label=="person":
		label = "P"+str(i)
		cv2.rectangle(orig_img,tuple(x[1:3].int()),tuple(x[3:5].int()),(0,255,0))
		font_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]  
		txtcorner = tuple(x[1:3].int())
		txtcorner = txtcorner[0]+font_sz[0]+2,txtcorner[1]-font_sz[1]-2
		cv2.rectangle(orig_img,tuple(x[1:3].int()),txtcorner,(255,0,0))
		cv2.putText(orig_img,label,(x[1].int(),x[2].int()), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

def crop_person(x,orig_img):
	return orig_img[x[2].int():x[4].int(),x[1].int():x[3].int()]

def track_person(TE,output,orig_img,tt=0.84):
	maxSim = 0
	maxEmbed = TE
	for x in output:	
		label = class_names[int(x[-1])]
		if label=="person":
			PE = img2vec.get_vec(crop_person(x,orig_img))
			sim = cosine_similarity(TE.reshape((1, -1)), PE.reshape((1, -1)))[0][0]
			if sim>tt:
				label = str(sim)
				cv2.rectangle(orig_img,tuple(x[1:3].int()),tuple(x[3:5].int()),(255,0,0))
				font_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]  
				txtcorner = tuple(x[1:3].int())
				txtcorner = txtcorner[0]+font_sz[0]+2,txtcorner[1]+font_sz[1]+2,
				cv2.rectangle(orig_img,tuple(x[1:3].int()),txtcorner,(255,0,0))
				cv2.putText(orig_img,label,(x[1].int(),txtcorner[1]), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))
				if sim>maxSim:
					maxEmbed = PE
	return maxEmbed
	

if __name__=="__main__":
	args = parser()
	confthres = float(args["confthres"])
	nmsthres = float(args["nmsthres"])
	trackthres = float(args["trackthres"]) 

	CUDA = torch.cuda.is_available()
	device = torch.device("cuda" if CUDA else "cpu")

	num_classes = 80
	fclasses = open("./data/coco.names","r")
	class_names = fclasses.read().split("\n")[:-1]

	print("Loading YOLO RPN...")
	try:
		YOLO = DarknetTorch(args["cfg"])
		YOLO.loadWeights(args["weights"])
		print("Load success!")
	except:
		raise LoadError("Unable to load YOLO RPN...")

	print("Loading "+args["ivmod"]+" embedding network...")
	try:
		img2vec = Img2Vec(model = args["ivmod"],layer_output_size=2048)
		print("Load success!")
	except:
		raise LoadError("Unable to load "+args["ivmod"]+" embedding network...")	


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
		flTrack = False
		# trackEmbed = []
		print("********Press p to Select Person********")
		print("********Press c to Stop Tracking/Change Person********")
		print("********Press q to Quit********")

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

			orig_dim = orig_dim.repeat(output.size(0), 1)
			scaling_factor = torch.min(dim/orig_dim,1)[0].view(-1,1)

			output[:,[1,3]] -= (dim - scaling_factor*orig_dim[:,0].view(-1,1))/2
			output[:,[2,4]] -= (dim - scaling_factor*orig_dim[:,1].view(-1,1))/2

			output[:,1:5] /= scaling_factor

			for i in range(output.shape[0]):
				output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, orig_dim[i,0])
				output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, orig_dim[i,1])

			if flTrack:
				trackEmbed = track_person(trackEmbed,output,orig_img,trackthres)
			else:
				list(map(lambda x,i: mark_all_persons(x,orig_img,i),output,range(len(output))))
			
				

			#show video frame
			cv2.imshow("frame", orig_img)
			key = cv2.waitKey(1)

			if key & 0xFF == ord('q'):
				break
			
			elif key & 0xFF == ord('p'):
				ID = int(input("Enter ID: "))
				print("Starting Tracking")
				trackEmbed = img2vec.get_vec(crop_person(output[ID],orig_img))
				flTrack = True
			
			elif key & 0xFF == ord('c'):
				print("Stopping Tracking")
				flTrack = False

			currFrame += 1
			print("FPS of the video is %f."%( currFrame / (time.time() - start)))


