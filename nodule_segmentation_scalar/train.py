from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import torch.utils.data
import torch
import torch.nn.functional as F
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch.nn
import matplotlib.pyplot as plt
from dataset import LidcDataLoader
import torchsummary
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, threshold_predictions_v, AverageMeter, calc_loss_scaler
from ploting import plot_kernels, LayerActivations, input_images
from Metrics import iou_score, dice_coef, dice_coef_scaler
import time, datetime
from inputimeout import inputimeout
from torch.cuda.amp import autocast, GradScaler
def get_input_with_timeout(prompt, timeout=10, default_value=None):
    try: 
        time_over = inputimeout(prompt=prompt, timeout=timeout) 
    except Exception: 
        time_over = default_value
    return time_over

def config_params():
    config = {
    'epoch': 60,
    'batch_size': 4,
    'num_workers': 16,
    'random_seed': random.randint(1, 5000),
    'accumulation_steps':  4,
    'initial_lr' : 0.00001,
    'weight_decay' : 1e-4
    }
    return config

def config_log():
    config = {'code': random.randint(1, 5000),
              'epoch':0,
              'lr':0.001}
    return config

def datalocation():
    folderlist = {
    't_data': 'E:/data/npy/Image/',
    'l_data': 'E:/data/npy/Mask/',
    'test_image': 'E:/data/npy/Clean/Image/0028_CN001_slice000.npy',
    'test_label': 'E:/data/npy/Clean/Mask/0028_CM001_slice000.npy',
    'test_folderP': 'E:\data\npy\Clean\Image\*',
    'test_folderL': 'E:\data\npy\Clean\Mask\*',
    'meta_file': 'E:/data/npy/meta.csv'
    }
    return folderlist

def print_config(config):
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

def train(model, train_loader, optimizer, n_iter, epoch, train_loss):
        model.train()
        k = 1
               
        avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
        pbar = tqdm(total=len(train_loader))      
        for input, target in train_loader:
            input, target = input.cuda(), target.cuda()
            input_images(input, target, epoch, n_iter, k)
            optimizer.zero_grad()
            output = model(input)
            lossT = calc_loss(output, target)     # Dice_loss Used
            dice = dice_coef(output, target)
            iou = iou_score(output, target)
            train_loss += lossT.item() * input.size(0)
            lossT.backward()
            optimizer.step()
            avg_meters['loss'].update(lossT.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
            k = 2
            pbar.close()
        return postfix

def train_acc(model, train_loader, optimizer, n_iter, epoch, train_loss, config):
    model.train()
    k = 1
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    pbar = tqdm(total=len(train_loader))
    accumulation_steps_count = 0
    
    for input, target in train_loader:
        input, target = input.cuda(), target.cuda()
        input_images(input, target, epoch, n_iter, k)

        optimizer.zero_grad()
        output = model(input)
        lossT = calc_loss(output, target)  # Dice_loss Used
        dice = dice_coef(output, target)
        iou = iou_score(output, target)

        #train_loss += lossT.item() * input.size(0)
        lossT.backward()

        accumulation_steps_count += 1

        if accumulation_steps_count == config['accumulation_steps']:
            # Perform optimizer step after accumulation_steps batches
            optimizer.step()
            optimizer.zero_grad()
            accumulation_steps_count = 0

        avg_meters['loss'].update(lossT.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])

        pbar.set_postfix(postfix)
        pbar.update(1)
        k = 2

    pbar.close()
    return postfix

def train_acc_time(model, train_loader, optimizer, n_iter, epoch, train_loss, config):
    model.train()
    k = 1
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'batch_time' : AverageMeter(),
                  'data_time' : AverageMeter()}
    pbar = tqdm(total=len(train_loader))
    accumulation_steps_count = 0
    end = time.time()
    scaler = GradScaler()
    for input, target in train_loader:
        # measure data loading time
        avg_meters['data_time'].update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        input_images(input, target, epoch, n_iter, k)

        optimizer.zero_grad()
        
        with autocast():  
            output = model(input)          
            lossT = calc_loss(output, target)  # Dice_loss Used
            scaler.scale(lossT).backward()
            dice = dice_coef(output, target)
            iou = iou_score(output, target)
        accumulation_steps_count += 1

        if accumulation_steps_count == config['accumulation_steps']:
            # Perform optimizer step after accumulation_steps batches
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            avg_meters['batch_time'].update(time.time() - end)
            end = time.time()            
            accumulation_steps_count = 0

        avg_meters['loss'].update(lossT.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('batch_time', avg_meters['dice'].avg),
            ('data_time', avg_meters['dice'].avg),
        ])

        pbar.set_postfix(postfix)
        pbar.update(1)
        k = 2
       
    pbar.close()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return postfix

def validation(model, valid_loader, valid_loss):
    model.eval()

    with torch.no_grad():
        avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
        pbar = tqdm(total=len(valid_loader))
        for input, target in valid_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            lossL = calc_loss(output, target)  # Dice_loss Used
            dice = dice_coef(output, target)
            iou = iou_score(output, target)            
            avg_meters['loss'].update(lossL.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))            
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            valid_loss += lossL.item() * input.size(0)
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return postfix

def log_execution(log, train, validation, param_log):    
    tmp = pd.Series(
            [param_log['code'],
            param_log['epoch'],
            param_log['lr'],
            train['loss'],
            train['iou'],
            train['dice'],
            validation['loss'],
            validation['iou'],
            validation['dice']],
            index=['code','epoch', 'lr', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice'])
    log = log._append(tmp, ignore_index=True)
    return log

def log_save(log):
    log.to_csv('./model/log.csv', index=False)

def train_folders(config):
    #######################################################
    #Creating a Folder for every data of the program
    #######################################################

    New_folder = './model'

    if os.path.exists(New_folder) and os.path.isdir(New_folder):
        shutil.rmtree(New_folder)

    try:
        os.mkdir(New_folder)
    except OSError:
        print("Creation of the main directory '%s' failed " % New_folder)
    else:
        print("Successfully created the main directory '%s' " % New_folder)

    #######################################################
    #Setting the folder of saving the predictions
    #######################################################

    read_pred = './model/pred'

    #######################################################
    #Checking if prediction folder exixts
    #######################################################

    if os.path.exists(read_pred) and os.path.isdir(read_pred):
        shutil.rmtree(read_pred)

    try:
        os.mkdir(read_pred)
    except OSError:
        print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
    else:
        print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

    #######################################################
    #checking if the model exists and if true then delete
    #######################################################

    read_model_path = './model/Unet_D_' + str(config['epoch']) + '_' + str(config['batch_size'])

    if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
        shutil.rmtree(read_model_path)
        print('Model folder there, so deleted for newer one')

    try:
        os.mkdir(read_model_path)
    except OSError:
        print("Creation of the model directory '%s' failed" % read_model_path)
    else:
        print("Successfully created the model directory '%s' " % read_model_path)

def test_folders():
    #######################################################
    #opening the test folder and creating a folder for generated images
    #######################################################

    read_test_folder112 = './model/gen_images'


    if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
        shutil.rmtree(read_test_folder112)

    try:
        os.mkdir(read_test_folder112)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder112)
    else:
        print("Successfully created the testing directory %s " % read_test_folder112)


    #For Prediction Threshold

    read_test_folder_P_Thres = './model/pred_threshold'


    if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
        shutil.rmtree(read_test_folder_P_Thres)

    try:
        os.mkdir(read_test_folder_P_Thres)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
    else:
        print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

    #For Label Threshold

    read_test_folder_L_Thres = './model/label_threshold'


    if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
        shutil.rmtree(read_test_folder_L_Thres)

    try:
        os.mkdir(read_test_folder_L_Thres)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
    else:
        print("Successfully created the testing directory %s " % read_test_folder_L_Thres)

def test(model, test_loader):
    test_folders()
    #######################################################
    #saving the images in the files
    #######################################################

    img_test_no = 0
    i = 0
    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter()}
 
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])

            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            output = np.squeeze(output, axis=1)
            pred = threshold_predictions_v(output)
            labl = target.cpu().numpy()
            
            np.save(f'./model/gen_images/output_{i}_img_no_{img_test_no}.npy', output)
            np.save(f'./model/pred_threshold/pred_tr_{i}_img_no_{img_test_no}.npy', pred)
            np.save(f'./model/label_threshold/lbl_tr_{i}_img_no_{img_test_no}.npy', labl)
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    print("="*50)
    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print('DICE:{:.4f}'.format(avg_meters['dice'].avg))
    print("="*50)    

def main():
    #######################################################
    #Checking if GPU is used
    #######################################################
    timer_count = 0
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    #######################################################
    #Setting the basic paramters of the model
    #######################################################
    config = config_params()    
    print_config(config)

    valid_loss_min = np.Inf
    lossT = []
    lossL = []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = config['epoch']-2
    n_iter = 1
    i_valid = 0

    pin_memory = False
    if train_on_gpu:
        pin_memory = True

    #######################################################
    #Setting up the model
    #######################################################

    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]

    def model_unet(model_input, in_channel=1, out_channel=1):
        model_test = model_input(in_channel, out_channel)
        return model_test

    model_test = model_unet(model_Inputs[0], 1, 1)

    model_test.to(device)
    model_test = nn.DataParallel(model_test)

    #######################################################
    #Getting the Summary of Model
    #######################################################

    torchsummary.summary(model_test, input_size=(1, 128, 128))

    #######################################################
    #Passing the Dataset of Images and Labels
    #######################################################
    folderslist = datalocation()
    t_data = folderslist['t_data']
    l_data = folderslist['l_data']
    test_image = folderslist['test_image']
    test_label = folderslist['test_label']
   
    meta = pd.read_csv(folderslist['meta_file'])  
    meta['original_image'] = meta['original_image'].apply(lambda x: t_data + '/' + x + '.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:l_data+ '/' + x +'.npy')    
    train_meta = meta[meta['data_split']=='Train']
    val_meta = meta[meta['data_split']=='Validation']
    test_meta = meta[meta['data_split']=='Test']
    
    # Get all *npy images into list for Train
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])
    # Get all *npy images into list for Validation
    val_image_paths = list(val_meta['original_image'])
    val_mask_paths = list(val_meta['mask_image'])
    # Get all *npy images into list for Test
    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])

    print("*"*50)
    print("The lenght of image: {}, mask folders: {} for train".format(len(train_image_paths),len(train_mask_paths)))
    print("The lenght of image: {}, mask folders: {} for validation".format(len(val_image_paths),len(val_mask_paths)))
    print("The lenght of image: {}, mask folders: {} for test".format(len(test_image_paths),len(test_mask_paths)))
    print("Ratio between Val/ Train is {:2f}".format(len(val_image_paths)/len(train_image_paths)))
    print("Ratio between Test/ Train is {:2f}".format(len(test_image_paths)/len(train_image_paths)))
    print("*"*50)

    #    Training_Data = LidcDataLoader(t_data, l_data)
    train_dataset = LidcDataLoader(train_image_paths, train_mask_paths)
    val_dataset = LidcDataLoader(val_image_paths,val_mask_paths)
    test_dataset = LidcDataLoader(test_image_paths,test_mask_paths)

    #######################################################
    #Giving a transformation for input data
    #######################################################

    data_transform = albu.Compose([
            albu.Rotate(limit=30, p=0.5),  # Rotate by up to 80 degrees
            albu.HorizontalFlip(p=0.15), # Horizontal flip
            albu.ElasticTransform(alpha=1.1, alpha_affine=0.5, sigma=5, p=0.5),  # Elastic transform
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),  # Shift, scale, and rotate
            albu.Affine(shear=[-15,15], p=0.5,translate_px=(-15, 15)),
            ToTensorV2()
        ])

    #######################################################
    #Trainging Validation Loader
    #######################################################

    print("Creating Dataloader Train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                            num_workers=config['num_workers'], pin_memory=pin_memory,)
    print("Creating Dataloader Validation")
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'],
                                            num_workers=config['num_workers'], pin_memory=pin_memory,)

    #######################################################
    #Using Adam as Optimizer
    #######################################################

    params = filter(lambda p: p.requires_grad, model_test.parameters())

    optimizer = torch.optim.Adam(params, lr=config['initial_lr'], weight_decay=config['weight_decay'])

    log = pd.DataFrame(index=[],columns= ['code','epoch', 'lr', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice'])

    #######################################################
    #Training loop
    #######################################################
    train_folders(config)
    best_dice = 0
    early_counter = 0
    param_log = config_log()
    for i in range(config['epoch']):

        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()

        #train_res = train(model_test, train_loader, optimizer, n_iter, config['epoch'], train_loss)
        train_res = train_acc_time(model_test, train_loader, optimizer, n_iter, config['epoch'], train_loss, config)
        valid_res = validation(model_test, valid_loader, valid_loss)
        param_log['epoch'] = i
        log = log_execution(log, train_res, valid_res, param_log)
        #######################################################
        #Saving the predictions
        #######################################################
        im_tb = np.load(test_image)
        im_label = np.load(test_label)
        augmented = data_transform(image=im_tb, mask=im_label)
        s_tb = augmented['image']
        s_label = augmented['mask']
        s_label= s_label.reshape([1,512,512])

        with torch.no_grad():
            s_tb = s_tb.unsqueeze(0).to(device, dtype=torch.float32)
            pred_tb = model_test(s_tb)
            pred_tb = F.sigmoid(pred_tb)

        pred_tb_cpu = pred_tb.cpu().numpy()
        x1 = plt.imsave(
            './model/pred/img_iteration_' + str(n_iter) + '_epoch_'
            + str(i) + '.png', pred_tb_cpu[0, 0, :, :])

        train_loss = train_res['loss'] / len(train_dataset)
        valid_loss = valid_res['loss'] / len(val_dataset)

        if (i+1) % 1 == 0:
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tDICE:{:.6f}'.format(i + 1, config['epoch'], train_loss,
                                                                                        valid_loss, valid_res['dice']))
        early_counter += 1
        if valid_res['dice'] > best_dice:
            torch.save(model_test.state_dict(),'./model/Unet_D_' +
                       str(config['epoch']) + '_' + str(config['batch_size']) + '/Unet_epoch_' + str(config['epoch'])
                       + '_batchsize_' + str(config['batch_size']) + '.pth')
            best_dice = valid_res['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            early_counter = 0

        #timer_count += 1
        if timer_count == 5:
            timer_count = 0
            hours = get_input_with_timeout("Enter the number of hours to sleep (press Enter for default): ", timeout=10, default_value=0)
            seconds = int(hours) * 3600  # Convert hours to seconds
            print(f"Sleeping for {hours} hours...")
            time.sleep(seconds)
            print("Wake up!")

        #######################################################
        #Early Stopping
        #######################################################
        """
        if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:
            if round(valid_loss, 4) == round(valid_loss_min, 4):
                print(i_valid)
                i_valid = i_valid+1
            valid_loss_min = valid_loss
            #if i_valid ==3:
            #    break
        """
        #######################################################
        # Extracting the intermediate layers
        #######################################################

        #####################################
        # for kernals
        #####################################
        x1 = torch.nn.ModuleList(model_test.children())

        #####################################
        # for images
        #####################################
        x2 = len(x1)
        dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer

        img = np.load(test_image)
        augmented = data_transform(image=img, mask=im_label)
        s_tb = augmented['image']

        with torch.no_grad():
            s_tb = s_tb.unsqueeze(0).to(device, dtype=torch.float32)
            pred_tb = model_test(s_tb)
            pred_tb = F.sigmoid(pred_tb)

        plot_kernels(dr.features, n_iter, 7, cmap="rainbow")

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        n_iter += 1
    log_save(log)
    print('Finished Training')
    #######################################################
    #Loading the model
    #######################################################
    print('Starting Test')
    print("Creating Dataloader Test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                            num_workers=config['num_workers'], pin_memory=pin_memory,)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_test.load_state_dict(torch.load('./model/Unet_D_' +
                    str(config['epoch']) + '_' + str(config['batch_size'])+ '/Unet_epoch_' + str(config['epoch'])
                    + '_batchsize_' + str(config['batch_size']) + '.pth'))

    model_test.eval()    

    test(model_test, test_loader)    

if __name__ == '__main__':
    main()