import os
import time

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from model import SeeInDark

parent_dir = 'pytorch-Learning-to-See-in-the-Dark/'

input_dir = parent_dir + 'dataset/Sony/short/'
gt_dir = parent_dir + 'dataset/Sony/long/'
m_path = parent_dir + 'saved_model/'
m_name = 'checkpoint_sony_e4000.pth'
result_dir = parent_dir + 'test_result_Sony/'

device = torch.device('cuda')

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

model = SeeInDark()
model.load_state_dict(torch.load( m_path + m_name ,map_location={'cuda:1':'cuda:0'}))
model = model.to(device)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

def process_file(tuple_):
    file_, k = tuple_
    print("File {}/{}".format(k+1, lenL))
    file_used = file_.split('/')[-1]
    gt_file = file_used.split('_')[0] + '_00_10s.ARW'
    print(gt_file)
    gt_path = glob.glob(gt_dir + gt_file)[0]
    in_exposure = file_used[9:-5] # float(in_fn[9:-5])
    gt_exposure = gt_file[9:-5] # float(gt_fn[9:-5])
    ratio = min(float(gt_exposure) / float(in_exposure), 300)

    raw = rawpy.imread(file_)
    im = raw.raw_image_visible.astype(np.float32) 
    input_full = np.expand_dims(pack_raw(im),axis=0) *ratio

    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)	

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

    input_full = np.minimum(input_full,1.0)

    in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
    out_img = model(in_img)
    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

    output = np.minimum(np.maximum(output,0),1)

    output = output[0,:,:,:]
    gt_full = gt_full[0,:,:,:]
    scale_full = scale_full[0,:,:,:]
    origin_full = scale_full
    scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
    
    Image.fromarray((origin_full*255).astype('uint8')).save(result_dir + file_used + '_ori.png')
    Image.fromarray((output*255).astype('uint8')).save(result_dir + file_used + '_out.png')
    Image.fromarray((scale_full*255).astype('uint8')).save(result_dir + file_used + '_scale.png')
    Image.fromarray((gt_full*255).astype('uint8')).save(result_dir + file_used + '_gt.png')

import glob
import multiprocessing as mp

L = glob.glob(input_dir + '*')
lenL = len(L)

for i in range(lenL):
    process_file((L[i], i+1))

"""nb_cores = 20
pool = mp.Pool(nb_cores)
# all the files
pool.map(process_file, ((file_, i) for i, file_ in enumerate(L)))
pool.close()
pool.join()"""