from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

import numpy as np
import cv2 


def parse_Darknet_cfg(cfgFile):
	"""
	Takes darknet yolo.cfg and converts the text to a list of dictionaries
	representing various sections of the yolo network.
	"""
	assert cfgFile.endswith('.cfg'), '{} is not a .cfg file'.format(cfgFile)
	section = {}
	sections = []

	with open(cfgFile,'r') as fin:
		for line in fin:
			if line[0]=='#' or not line.lstrip().rstrip():
				pass
			else:
				if line.startswith('['):
					if section:		#append old section and initialize new one
						sections.append(section)
						section = {}
					section["type"] = line.strip().strip('[]')
				else:
					key,value = line.rstrip().split("=")
					section[key.rstrip()] = value.lstrip()
		sections.append(section)
		return sections


# Define custom module class for route concats and shortcut adds
# Just a dummy module, used to mark the above layers in the model definition
class DummyLayer(nn.Module):
	def __init__(self):
		super(DummyLayer, self).__init__()

# Define custom module class for yolo detection. 
# Just a dummy module to store anchors while initializing net.
class YOLODummyLayer(nn.Module):
	def __init__(self, anchors):
		super(YOLODummyLayer, self).__init__()
		self.anchors = anchors


def predictYOLO(pred, imgsz, anchors, num_classes = 80, CUDA = True):
	batch_sz = pred.size(0)
	stride =  imgsz // pred.size(2)
	grid_sz = imgsz // stride
	num_anchors = len(anchors)  #3

	#scale anchor boxes
	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	#transform pred tensor into output form
	pred = pred.view(batch_sz, (5 + num_classes)*num_anchors, grid_sz*grid_sz)
	pred = pred.transpose(1,2).contiguous() #view can work only on memory contiguous tensors
	pred = pred.view(batch_sz, grid_sz*grid_sz*num_anchors, (5 + num_classes))

	#tx, ty, pc are sigmoided and center offsets are added
	pred[:,:,0] = torch.sigmoid(pred[:,:,0])
	pred[:,:,1] = torch.sigmoid(pred[:,:,1])
	pred[:,:,4] = torch.sigmoid(pred[:,:,4])

	grid_len = np.arange(grid_sz)
	a,b = np.meshgrid(grid_len, grid_len)
	x_offset = torch.FloatTensor(a).view(-1,1)
	y_offset = torch.FloatTensor(b).view(-1,1)
	if CUDA:
		x_offset = x_offset.cuda()
		y_offset = y_offset.cuda()
	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
	pred[:,:,:2] += x_y_offset

	#scale anchors
	anchors = torch.FloatTensor(anchors)
	if CUDA:
		anchors = anchors.cuda()
	anchors = anchors.repeat(grid_sz*grid_sz, 1).unsqueeze(0)
	pred[:,:,2:4] = torch.exp(pred[:,:,2:4])*anchors

	#sigmoid class scores
	pred[:,:,5:5+num_classes] = torch.sigmoid((pred[:,:, 5:5+num_classes]))

	pred[:,:,:4] *= stride

	return pred


def IOU(box1, box2):
	"""
	Calculates the IoU of two bounding boxes 
	"""

	#Get bounding box corners
	b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
	b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

	#Intersection(n) corners
	nx1 =  torch.max(b1_x1, b2_x1)
	ny1 =  torch.max(b1_y1, b2_y1)
	nx2 =  torch.min(b1_x2, b2_x2)
	ny2 =  torch.min(b1_y2, b2_y2)

	#Intersection(n) area
	n_area = torch.clamp(nx2-nx1, min=0)*torch.clamp(ny2-ny1, min=0)

	#Rectangle Areas 
	b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
	b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)

	# IOU = Intersection Over Union = Intersection/Union
	iou = n_area / (b1_area + b2_area - n_area)

	return iou

def nmsYOLO(pred, device, confthres = 0.5, num_classes = 80, nmsthres = 0.4):
	"""
	Input: (batchsz x bounding boxes x bbox attributes) torch tensor
	Output: (nms bounding boxes x 8) torch tensor
	8 values: Batch Idx(0),bounding box corner coordinates (1,2,3,4),
			  object conf(5), max class prob (6), object label (7)
	"""
	batch_sz = pred.size(0)

	conf_mask = (pred[:,:,4]>confthres).float().unsqueeze(2)
	
	# Return if there is no confident prediction.
	if len(torch.nonzero(conf_mask))==0:
		return 0

	pred = pred*conf_mask
	# nzpred = torch.nonzero(pred[:,:,4])
	# pred = pred[]
	boxenlarge = 0.05
	pred[:,:,2:4] *= (boxenlarge+torch.exp((1-pred[:,:,4])/2).float().unsqueeze(2))

	# using center and box lengths, get topleft and bottomright corners
	# which will be easier to plot using opencv rectangle functionN
	corners = pred.new(pred[:,:,:4].shape)
	corners[:,:,0] = (pred[:,:,0] - pred[:,:,2]/2)
	corners[:,:,1] = (pred[:,:,1] - pred[:,:,3]/2)
	corners[:,:,2] = (pred[:,:,0] + pred[:,:,2]/2) 
	corners[:,:,3] = (pred[:,:,1] + pred[:,:,3]/2)
	pred[:,:,:4] = corners[:,:,:4]

	output = []
	# loop over all images in the given batch to get nms scores and new predictions
	for ib in range(batch_sz):
		imgPred = pred[ib]

		maxConf, maxClass= torch.max(imgPred[:,5:5+num_classes],1)
		maxConf = maxConf.float().unsqueeze(1)
		maxClass = maxClass.float().unsqueeze(1)
		imgPred = torch.cat((imgPred[:,:5],maxConf,maxClass),1)

		# Select only non-zero predictions
		nzpos =  (torch.nonzero(imgPred[:,4])).squeeze()
		imgPred = imgPred[nzpos,:].view(-1,7)

		try:
			npPred = imgPred[:,-1].cpu().numpy()
			predClasses = torch.from_numpy(np.unique(npPred))
			predClasses = predClasses.to(device)
		except:
			continue


		# Perform nms wrt person class and remove all other BBs
		PERSON = 0.0
		for class_ in predClasses:
			if PERSON != class_.cpu().numpy():
				continue
			else:
				predPerson = imgPred*(class_==imgPred[:,-1]).float().unsqueeze(1)
				nzmask = torch.nonzero(predPerson[:,-2]).squeeze()

				imgPred = imgPred[nzmask].view(-1,7)

				# sort detections in descending objectness confidence(4)
				confIdx = torch.sort(imgPred[:,4], descending=True)[1]
				imgPred = imgPred[confIdx]

				for i in range(imgPred.size(0)):
					try:
						iou = IOU(imgPred[i].unsqueeze(0),imgPred[i+1:])
					except:
						break

					ioumask = (iou < nmsthres).float().unsqueeze(1)
					imgPred[i+1:] = imgPred[i+1:]*ioumask
					nzmask = torch.nonzero(imgPred[:,4]).squeeze()
					imgPred = imgPred[nzmask].view(-1,7)


				batchIdx = imgPred.new(imgPred.size(0), 1).fill_(ib)

				out = torch.cat((batchIdx,imgPred),1)
				if isinstance(output, list):
					output = out
				else:
					output	= torch.cat((output,out))
				# print(out[:,5:7])
				break	#No need to check any other classes
	if isinstance(output,list):
		return 0
	return output


def module_define_torch(sections):
	"""
	Takes the sections dictionary and creates torch modules for the network
	"""
	moduleList = nn.ModuleList()
	m_idx = 0
	allfilters = [int(sections[0]["channels"])]	#cache to use in concatenation filter count for route layer

	for L in sections:
		module = nn.Sequential()

		if (L["type"] == "net"):
			continue

		elif (L["type"] == "convolutional"):
			activation = L["activation"]
			filters = int(L["filters"])
			size = int(L["size"])
			stride = int(L["stride"])
			pad = int(L["pad"])
			bn = 'batch_normalize' in L
			bias = not bn

			if pad:
				padding = (size - 1) // 2	#SAME Padding
			else:
				padding = 0

			#using functions predefined in nn module
			conv = nn.Conv2d(allfilters[-1], filters, size, stride, padding, bias = bias)
			module.add_module("conv_{}".format(m_idx),conv)
			if bn:
				BN = nn.BatchNorm2d(filters)
				module.add_module("batch_norm_{0}".format(m_idx), BN)

			if activation == "leaky":
				act = nn.LeakyReLU(0.1,inplace=True)
				module.add_module("leaky_{}".format(m_idx), act)

		elif (L["type"] == "route"):
			routeL = DummyLayer()
			module.add_module("route_{}".format(m_idx), routeL)
			
			# update filters/channels of this layer
			L["layers"] = L["layers"].split(',')
			# can be -4 for yolo branch route and -1 for concat route
			idxFrom = int(L["layers"][0])	
			try:	# concat route
				idxTill = int(L["layers"][1]) - m_idx	# layer number to take concat from;
				filters = allfilters[m_idx + idxFrom+1] + allfilters[m_idx + idxTill+1]
			except:	# yolo branch route
				filters = allfilters[m_idx + idxFrom+1]

		elif (L["type"] == "shortcut"):
			shortcut = DummyLayer()
			module.add_module("shortcut_{}".format(m_idx), shortcut)

		elif (L["type"] == "upsample"):
			stride = int(L["stride"])
			upsample = nn.Upsample(scale_factor = stride)
			module.add_module("upsample_{}".format(m_idx), upsample)
			
		elif (L["type"] == "yolo"):
			mask = [int(m) for m in L["mask"].split(",")]
			anchors = [int(a) for a in L["anchors"].split(",")]
			anchors = [(anchors[2*i],anchors[2*i+1]) for i in mask]

			yoloDetect = YOLODummyLayer(anchors)
			module.add_module("YOLODetect_{}".format(m_idx), yoloDetect)

		moduleList.append(module)
		allfilters.append(filters)
		m_idx+=1

	return(moduleList)


class DarknetTorch(nn.Module):
	def __init__(self,cfgFile):
		super(DarknetTorch, self).__init__()
		self.sections = parse_Darknet_cfg(cfgFile)	#list of dictionaries
		self.moduleList = module_define_torch(self.sections)
		self.header = torch.IntTensor([0,0,0,0])
		self.netparams = self.sections[0]		#dictionary defining net params


	def forward(self,x,CUDA):
		"""
		Takes the sections dictionary and builds torch modules for the network
		"""
		detects = []
		outputs = {}	#cache to use in concatenation filter count for route layer 

		for i,L in enumerate(self.sections[1:]):

			if (L["type"] == "net"):
				continue

			elif L["type"] == "convolutional" or L["type"] == "upsample":
				x = self.moduleList[i](x)
				outputs[i] = x

			elif L["type"] == "shortcut":
				shortcutFrom = int(L["from"])	#add this layers output to prev layer
				x = outputs[i-1] + outputs[i+shortcutFrom]
				outputs[i] = x

			elif (L["type"] == "route"):
				# can be -4 for yolo branch route and -1 for concat route
				idxFrom = int(L["layers"][0])	
				try:	# concat route
					idxTill = int(L["layers"][1]) - i	# layer number to take concat from;
					fmap1 = outputs[i + idxFrom] 
					fmap2 = outputs[i + idxTill]
					x = torch.cat((fmap1,fmap2),1)	# concatenate the two feature maps along the channel depth
				except:	# yolo branch route
					x = outputs[i + idxFrom]		# take branch from previous layer 
				outputs[i] = x
				
			elif (L["type"] == "yolo"):	
				anchors = self.moduleList[i][0].anchors
				# print(anchors)

				x = x.data
				x = predictYOLO(x, imgsz=int(self.netparams["height"]), anchors=anchors, num_classes=int(L["classes"]), CUDA=CUDA)
				if isinstance(detects, list):
					detects = x
				else:
					detects	= torch.cat((detects,x),1)	#concatenate along channel depth
				outputs[i] = outputs[i-1]
		# print(detects)
		return detects


	def loadWeights(self,wPath):
		assert wPath.endswith('.weights'), '{} is not a .weights file'.format(wPath)
		with open(wPath,"rb") as fw:
			# Load header information- Major version number, Minor Version Number
			# Subversion number, IMages seen 
			self.weights_header = np.ndarray(shape=(5, ), dtype='int32', buffer=fw.read(20))
			# print('Weights Header: ', self.weights_header)

			# load other weights
			weights = np.fromfile(fw, dtype = np.float32)
			ptr = 0
			for i,L in enumerate(self.sections[1:]):
				if L["type"] == "convolutional":
					module = self.moduleList[i]
					conv = module[0]
					if 'batch_normalize' in L:
						bn = module[1]

						filters = bn.bias.numel()
						W_bnbias = torch.from_numpy(weights[ptr:ptr+filters]) # shift beta
						ptr+=filters
						W_bnweights = torch.from_numpy(weights[ptr:ptr+filters])	# scale gamma
						ptr+=filters
						W_bnrunningmean = torch.from_numpy(weights[ptr:ptr+filters])	#running mean
						ptr+=filters
						W_bnrunningvar = torch.from_numpy(weights[ptr:ptr+filters])	#running var
						ptr+=filters

						#Copy the data to model as per required shape
						bn.bias.data.copy_(W_bnbias.view_as(bn.bias.data))
						bn.weight.data.copy_(W_bnweights.view_as(bn.weight.data))
						bn.running_mean.copy_(W_bnrunningmean.view_as(bn.running_mean))
						bn.running_var.copy_(W_bnrunningvar.view_as(bn.running_var))
					else:
						filters = conv.bias.numel()
						W_convbias = torch.from_numpy(weights[ptr:ptr+filters]) # shift beta
						ptr+=filters
						conv.bias.data.copy_(W_convbias.view_as(conv.bias.data))
					
					nW = conv.weight.numel()
					W_convweights = torch.from_numpy(weights[ptr:ptr+nW])
					# print(W_convweights)
					ptr+=nW
					conv.weight.data.copy_(W_convweights.view_as(conv.weight.data))


	def get_sections(self):
		return self.sections

	def get_moduleList(self):
		return self.moduleList


# secs = parse_Darknet_cfg("./cfg/yolov3.cfg")
# module_define_torch(secs)

# def get_test_input(input_dim, CUDA):
#     img = cv2.imread("dog-cycle-car.png")
#     img = cv2.resize(img, (input_dim, input_dim)) 
#     img_ =  img[:,:,::-1].transpose((2,0,1))
#     img_ = img_[np.newaxis,:,:,:]/255.0
#     img_ = torch.from_numpy(img_).float()
#     img_ = Variable(img_)
    
#     if CUDA:
#         img_ = img_.cuda()
#     return img_

# CUDA = torch.cuda.is_available()
# print(CUDA)
# dn = DarknetTorch('cfg/yolov3.cfg')
# dn.loadWeights("yolov3.weights")
# inp = get_test_input(320, CUDA)
# print(inp)
# if CUDA:
#     dn.cuda()
# dn(inp,CUDA)
# dn.eval()