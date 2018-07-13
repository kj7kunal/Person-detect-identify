import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


class Img2Vec():

    def __init__(self, model='resnet34', layer_output_size=512):
        self.CUDA = False #torch.cuda.is_available()
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model)
        print("Using "+ model)

        if self.CUDA:
            self.model.cuda()

        # Set to testing mode
        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def _get_model_and_layer(self, model_name):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)  
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)  
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)  
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)  
        else:
            raise KeyError('Model %s was not found' % model_name)

        layer = model._modules.get('avgpool')

        return model, layer


    def get_vec(self, img):
        """ Get vector embedding from PIL image
        """
        if self.CUDA:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        else:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))

        # initialize feature vector
        my_embedding = torch.zeros(1,self.layer_output_size, 1, 1)

        # hook function to copy layer output (passes model/input/output)
        # when forward pass reaches this layer, hook function executed as interrupt
        # store copy of output data in my_embedding
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        # attach hook function to selected layer 
        h = self.extraction_layer.register_forward_hook(copy_data)

        # run model on image
        # feature outputs will be stored in my_embedding after forward pass
        self.model(image)
        
        # Detach function from layer
        h.remove()
        return my_embedding.numpy()

    