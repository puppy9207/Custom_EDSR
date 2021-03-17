import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from edsr import EDSR
from PIL import Image
import imageio
import os
import numpy as np
import time

def image_loader(image_name):
    """load image, returns cuda tensor"""
    #image read and to pytorch variable
    image = imageio.imread(image_name)
    np_transpose = np.ascontiguousarray(image.transpose((2, 0, 1)))
    image = torch.from_numpy(np_transpose).float()
    image = Variable(image, requires_grad=True)
    #Add Batch 1. Because the model achitecture is like that.
    image = image.unsqueeze(0)
    #I have GPU
    return image.cuda() 

#My image path
BEFORE_PATH = "./image/before/"
AFTER_PATH = './image/after/'
# model load
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EDSR().to(DEVICE)
# EDSR pre-trained model. It is 64 feats, scale 4 model
model.load_state_dict(torch.load("edsr_baseline_x4-6b446fab.pt"))
# model execute and save image
file_list = os.listdir(BEFORE_PATH)


start = time.time()
temp = []
print(file_list)
with torch.no_grad():
    for file in file_list:
        full_file = os.path.join(BEFORE_PATH+file)
        img = image_loader(full_file)
        output = model(img)
        #Adjust values above 255 to 255
        normalized = output.clamp(0, 255)
        normalized = normalized.squeeze()
        tensor_cpu = normalized.permute(1,2,0).byte().cpu()
        after_file = os.path.join(AFTER_PATH,file)
        temp.append([after_file,tensor_cpu.numpy()])
        imageio.imwrite(after_file, tensor_cpu.numpy())
print("time : {}".format(time.time()-start))