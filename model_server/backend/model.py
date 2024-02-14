import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torchvision import transforms
from datetime import datetime
from PIL import Image
from models.UTNet.utnet import UTNet
from models.AttentionUNet.AUNET import AttentionUNet
from models.Unet.unet_model import UNet
from imageio import imwrite
import matplotlib.image
import cv2

def save_output(output,output_directory):
    # This saves the predicted image into a directory. The naming convention will follow PI
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name = 'output_img_pred_' + formatted_datetime

    for i in range(output.shape[0]):
        np.save(output_directory+'/'+file_name,output[i,:,:])
        print("SAVED",file_name+'.npy')
        
    numpy_array = np.load(output_directory+'/'+file_name+'.npy')
    numpy_array = np.uint8(numpy_array)
    
    matplotlib.image.imsave(os.path.join(output_directory, file_name + '.png'), numpy_array, cmap='gray')
    print("SAVED", file_name + '.png')
    return os.path.abspath(os.path.join(output_directory, file_name + '.png'))

def predict(image_path):
    #image = np.load('test.npy')
    image = image_path
    transformations = transforms.Compose([transforms.ToTensor()])
    image = transformations(image)

    # load configuration
    with open('models/UTNet/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model")    
    #model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = UTNet(1, config['base_chan'], config['num_class'], reduce_size=config['reduce_size'], block_list=config['block_list'], num_blocks=config['num_blocks'], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config['aux_loss'], maxpool=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print("Loading model file from")
    #model.load_state_dict(torch.load('models/Unet/model.pth'))
    model.load_state_dict(torch.load('models/UTNet/model.pth'))
    model = model.cuda()

    model.eval()

    OUTPUT_DIR = 'output'

    input = image
    input = input.unsqueeze(0)  # Add batch dimension
    input = input.cuda()  
    
    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output)
        output = (output>0.5).float().cpu().numpy()
        output = np.squeeze(output,axis=1)
        object_png = save_output(output,OUTPUT_DIR)
    print(object_png)
    torch.cuda.empty_cache()
    return object_png


def main():
    image = np.load('test.npy')
    transformations = transforms.Compose([transforms.ToTensor()])
    image = transformations(image)

    # load configuration
    with open('models/UTNet/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model")    
    #model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = UTNet(1, config['base_chan'], config['num_class'], reduce_size=config['reduce_size'], block_list=config['block_list'], num_blocks=config['num_blocks'], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config['aux_loss'], maxpool=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print("Loading model file from")
    #model.load_state_dict(torch.load('models/Unet/model.pth'))
    model.load_state_dict(torch.load('models/UTNet/model.pth'))
    model = model.cuda()

    model.eval()

    OUTPUT_DIR = 'output'

    input = image
    input = input.unsqueeze(0)  # Add batch dimension
    input = input.cuda()  
    
    with torch.no_grad():
        output = model(input)
        output = torch.sigmoid(output)
        output = (output>0.5).float().cpu().numpy()
        output = np.squeeze(output,axis=1)
        object_png = save_output(output,OUTPUT_DIR)
    print(object_png)
    torch.cuda.empty_cache()
    return object_png 

if __name__ == '__main__':
    main()