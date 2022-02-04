import os
import numpy as np
import pandas as pd
import rawpy
import glob

import torch

from PIL import Image
from skimage.metrics import structural_similarity as ssim

from model import SeeInDark

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None,
    type=str, help="Tester la méthode sur un fichier particulier.")
parser.add_argument("--generalization", default=False,
    type=bool, help="Tester sur d'autres images provenant d'une autre trame de Bayer.")
parser.add_argument('--dataset_dir',
    type=str, default='', help="Chemin d'accès au dataset")
parser.add_argument('--model_dir',
    type=str, default='', help="Chemin d'accès au(x) modèle(s)")
parser.add_argument('--black_level', type=int,
    default=512, help="Niveau de noir : 512 (Sony), 528 (Iphone XR)")
args = parser.parse_args()

dataset_dir = args.dataset_dir
models_dir = args.model_dir
black_level = args.black_level

# chemin d'accès vers les images à débruiter (images courte-exposition)
input_dir = dataset_dir + 'dataset/Sony/short/'
# chemin d'accès vers les ground truth associées (images haute-exposition)
gt_dir = dataset_dir + 'dataset/Sony/long/'

# chemin d'accès où sont stockés les modèles .pth
m_path = models_dir + 'saved_model/'

# chemin d'accès pour stocker les sorties du modèle (images)
result_dir = dataset_dir + 'test_result_Sony/'

def pack_raw(raw, black_level):
    # pack Bayer image vers 4 canaux
    # soustraction du niveau de noir
    im = np.maximum(raw - black_level, 0)/ (16383 - black_level)

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def compute_ssim(gt_im, pred_im):
    """ Calcul de la SSIM entre l'image ground truth et l'image prédite.
    """
    return(ssim(gt_im, pred_im,
                  data_range=pred_im.max() - pred_im.min(), multichannel=True))

def process_file(tuple_, d: dict, name_dir: str):
    """ Fonction qui prédit l'image débruitée à partir de l'image donnée en entrée.
    """
    file_, k = tuple_
    print("Fichier {}/{}".format(k, lenL))
    file_used = file_.split('/')[-1]
    if not args.generalization:
        gt_path = glob.glob(gt_dir + file_used.split('_')[0] + "*")[0]
        gt_file = gt_path.split('/')[-1]
        in_exposure = file_used[9:-5]
        gt_exposure = gt_file[9:-5]
        # hyperparamètre gamma
        ratio = min(float(gt_exposure) / float(in_exposure), 300)
    else:
        ratio = 200

    raw = rawpy.imread(file_)
    im = raw.raw_image_visible.astype(np.float32)
    input_full = np.expand_dims(pack_raw(im, black_level),axis=0) * ratio

    if not args.generalization:
        # si l'on est dans le cas de l'appareil photo Sony,
        # on dispose de l'image Ground Truth...
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

        gt_full = gt_full[0,:,:,:]
        scale_full = scale_full[0,:,:,:]
        origin_full = scale_full

    input_full = np.minimum(input_full, 1.0)
    if args.generalization:
        # Cas Iphone XR --> redimensionner dimensions en multiple de 32 pour U-Net
        input_full = input_full[:, :1440, :1920, :]
    in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)

    out_img = model(in_img)
    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]

    if not args.generalization:
        # sortir les métriques de score sous format pandas
        # si l'on dispose de la ground truth dans le cas de l'appareil photo Sony
        d['id'].append(k)
        d['ground truth'].append(gt_file)
        d['predicted image'].append(file_used + '_out.png')
        d['ssim'].append(compute_ssim(gt_full*255, output*255))

        # sauvegarder la ground truth et l'image donnée en entrée
        Image.fromarray((origin_full*255).astype('uint8')).save(name_dir + file_used + '_ori.png')
        Image.fromarray((gt_full*255).astype('uint8')).save(name_dir + file_used + '_gt.png')
    # sauvegarder l'image de sortie (=image prédite)
    Image.fromarray((output*255).astype('uint8')).save(name_dir + file_used + '_out.png')


if __name__ == "__main__":
    # activer le mode GPU
    device = torch.device('cuda')
    # création du modèle
    model = SeeInDark()

    # récupérer tous les modèles .pth du dossier pour évaluation
    m_names = glob.glob(m_path + "*.pth")

    # boucle sur tous les checkpoints .pth
    for m_name in m_names:
        m_name = m_name.split('/')[-1]
        name_dir = m_name.split('.')[0] + '/'
        os.makedirs(name_dir, exist_ok=True)
        print('{}'.format(m_name))
        model.load_state_dict(torch.load(m_path + m_name, map_location={'cuda:1':'cuda:0'}))
        model = model.to(device)

        os.makedirs(result_dir, exist_ok=True)

        if not args.file:
            # évaluation sur l'ensemble du jeu de test
            L = glob.glob(input_dir + '1*')
        else:
            # évaluation sur un simple fichier donné en argument
            L = [input_dir + args.file]
        lenL = len(L)

        d = {"id": [], "ground truth": [], "predicted image": [], "ssim": []}
        for i in range(lenL):
            process_file((L[i], i+1), d, name_dir)
            # sauvegarder le dataframe avec les métriques en fichier .csv
            df = pd.DataFrame.from_dict(d)
            df.to_csv('test_results_{}.csv'.format(m_name))