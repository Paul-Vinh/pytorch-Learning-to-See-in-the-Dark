import os
import time

import numpy as np
import matplotlib.pyplot as plt
import rawpy
import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from model import SeeInDark

import argparse
from pytorch_msssim import SSIM
from torchgeometry.losses import ssim

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', type=str,
    default='', help='Chemin d'accès au dataset')
parser.add_argument('--models_dir', type=str,
    default='', help='Chemin pour stocker les modèles .pth')
parser.add_argument('--loss', type=str,
    default='L1', help="Loss d'entraînement = 'L1', 'L2' or 'ssim'")
parser.add_argument('--epoch', type=int,
    default=10, help="Nombre d'epochs")
parser.add_argument('--save_freq', type=int,
    default=100, help='Sauvegarder les poids tous les save_freq')
parser.add_argument('--val_freq', type=int,
    default=100, help='Tester le modèle sur le jeu de validation tous les val_freq')
parser.add_argument('--plot_loss', type=bool,
    default=False, help='Tracé de la loss ?')
parser.add_argument('--black_level', type=int,
    default=512, help="Niveau de noir : 512 (Sony), 528 (Iphone XR)")

args = parser.parse_args()

type_loss = args.loss
num_epochs = args.epoch
plot_loss = args.plot_loss
parent_dir = args.parent_dir

save_freq = args.save_freq
val_freq = args.val_freq

# chemin d'accès vers les images à débruiter (images courte-exposition)
input_dir = parent_dir + 'dataset/Sony/short/'
# chemin d'accès vers les ground truth associées (images haute-exposition)
gt_dir = parent_dir + 'dataset/Sony/long/'
# chemin pour stocker les résultats
result_dir = parent_dir + 'result_Sony/'
model_dir = args.models_dir

if not os.path.isdir(result_dir):
  os.mkdir(result_dir)

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)

# utiliser GPU
device = torch.device('cuda')
print(f"Device: {device}")

# récupération des images d'entraînement du dataset SID
# les fichiers commencent par '0'
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []

for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

# récupération du jeu de validation
val_fns = glob.glob(gt_dir + '/2*.ARW')
val_ids = []
for i in range(len(val_fns)):
    _, val_fn = os.path.split(val_fns[i])
    val_ids.append(int(val_fn[0:5]))

# taille du patch pour entraînement
ps = 512

DEBUG = 0
if DEBUG == 1:
    save_freq = 100
    train_ids = train_ids[0:5]
    val_ids = val_ids[0:5]

def pack_raw(raw, black_level):
    # pack Bayer image vers 4 canaux
    im = raw.raw_image_visible.astype(np.float32)
    # soustraction du niveau de noir
    im = np.maximum(im - 512,0)/ (16383 - black_level)

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def loss_l1(out_im, gt_im):
    """ Loss L1 d'entraînement.
    """
    return torch.abs(out_im - gt_im).mean()

def loss_l2(out_im, gt_im):
    """ Loss L2 d'entraînement.
    """
    loss = nn.MSELoss() 
    return loss(out_im, gt_im)

def loss_ssim(out_im, gt_im):
    """ Loss SSIM d'entraînement.
    """
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    return(1 - ssim_loss(out_im, gt_im))

g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

# hyperparamètres + modèle & optimiseur
learning_rate = 1e-4
model = SeeInDark().to(device)
model._initialize_weights()
opt = optim.Adam(model.parameters(), lr = learning_rate)

# évolution des métriques
train_loss = []
val_loss = []

# boucle sur le nombre d'epochs
for epoch in range(max(1, lastepoch), num_epochs+1):
    cnt = 0
    # si epoch > 2000 ==> le taux d'apprentissage est modifié
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5
    
    model.train()

    for ind in np.random.permutation(len(train_ids)):
        # avoir le chemin d'accès à partir de l'id de l'image
        train_id = train_ids[ind]
        # fichier donné en entrée (input)
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.randint(0,len(in_files))]
        _, in_fn = os.path.split(in_path)

        # ground truth
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        # facteur gamma : hyperparamètre qui contrôle l'éclairement
        ratio = min(gt_exposure / in_exposure, 300)
          
        st = time.time()
        cnt += 1

        raw = rawpy.imread(in_path)
        input_image = np.expand_dims(pack_raw(raw, args.black_level),axis=0) *ratio
    
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_image = np.expand_dims(np.float32(im/65535.0),axis = 0)
         
        # cropping de l'image
        H = input_image.shape[1]
        W = input_image.shape[2]

        xx = np.random.randint(0, W-ps)
        yy = np.random.randint(0, H-ps)
        input_patch = input_image[:, yy:yy+ps, xx:xx+ps,:]
        gt_patch = gt_image[:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]


        # Data Augmentation
        if np.random.randint(2,size=1)[0] == 1:
            # Random Flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2,size=1)[0] == 1:
            # Random Transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        # conversion des images (input + ground truth) en tenseur Pytorch
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)

        model.zero_grad()
        # prédiction sur l'image donnée en entrée
        out_img = model(in_img)
        
        # sélection de la loss d'entraînement
        if type_loss == "L1":
            loss = loss_l1(out_img, gt_img)
        elif type_loss == "L2":
            loss = loss_l2(out_img, gt_img)
        elif type_loss == "ssim":
            loss = loss_ssim(out_img, gt_img)
        loss.backward()

        opt.step()
        g_loss[ind] = loss.data.cpu()

        # Loss moyenne
        mean_loss = np.mean(g_loss[np.where(g_loss)])
        print(f"Training: Epoch: {epoch} \t Count: {cnt} \t Loss={mean_loss:.3} \t Time={time.time()-st:.3}")
        train_loss.append(mean_loss)

    # sauvegarde du modèle sous format .pth
    if epoch % save_freq == 0 or epoch == num_epochs:
        torch.save(model.state_dict(), model_dir+'checkpoint_sony_'+type_loss+'e%04d.pth'%epoch)

    # test sur 5 images du jeu de validation à chaque fois prises au hasard
    # dans le jeu de données pris en entier
    if epoch % val_freq == 0 or epoch == num_epochs:
        model.eval()
        loss_value = 0
        for ind in np.random.permutation(len(val_ids))[:5]:
            # récupérér le chemin à partir de l'id de l'image
            val_id = val_ids[ind]
            # image donnée en entrée
            in_files = glob.glob(input_dir + '%05d_00*.ARW'%val_id)
            in_path = in_files[np.random.randint(0,len(in_files))]
            _, in_fn = os.path.split(in_path)

            # ground truth
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%val_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure =  float(in_fn[9:-5])
            gt_exposure =  float(gt_fn[9:-5])
            # hyperparamètre gamma
            ratio = min(gt_exposure / in_exposure, 300)
            
            st = time.time()
            cnt += 1

            raw = rawpy.imread(in_path)
            input_image = np.expand_dims(pack_raw(raw, black_level),axis=0) *ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_image = np.expand_dims(np.float32(im/65535.0),axis = 0)
            
            in_img = torch.from_numpy(input_image).permute(0,3,1,2).to(device)
            gt_img = torch.from_numpy(gt_image).permute(0,3,1,2).to(device)

            out_img = model(in_img)
            
            # Choix de la loss (L1, L2, SSIM) pour l'étape de validation
            if type_loss == "L1":
                loss = loss_l1(out_img, gt_img)
            elif type_loss == "L2":
                loss = loss_l2(out_img, gt_img)
            elif type_loss == "ssim":
                loss = loss_ssim(out_img, gt_img)

            g_loss[ind] = loss.data.cpu()

            mean_loss = np.mean(g_loss[np.where(g_loss)])
            print(f"Validation : Epoch: {epoch} \t Count: {cnt} \t Loss={mean_loss:.3} \t Time={time.time()-st:.3}")
            loss_value += mean_loss
        val_loss.append(loss_value/5)

# si on souhaite tracer les courbes de loss
if plot_loss:
    # Loss d'entraînement + Loss de validation
    plt.plot([i/len(train_ids) for i in range(len(train_loss))], train_loss, label = 'train loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label = "val loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss ' + type_loss)
    plt.legend()
    # sauvegarder les courbes en fichier .png
    plt.savefig(model_dir+'Loss'+type_loss+'e%04d.png'%epoch)   