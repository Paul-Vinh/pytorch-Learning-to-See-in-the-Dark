# pytorch-Learning-to-See-in-the-Dark [Projet Imagerie Numérique (MVA 2021-2022)]
Learning to See in the Dark utilisant PyTorch 0.4.0 & 1.0.0

### Version Tensorflow originale
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018. <br/>
[Tensorflow code](https://github.com/cchen156/Learning-to-See-in-the-Dark) <br/>
[Paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)

### Github repository forked from https://github.com/lavi135246/pytorch-Learning-to-See-in-the-Dark
Nous nous sommes fortement inspirés du code de ce dossier Github fourni en version Pytorch. Toutefois, nous avons apporté des changements dans l'implémentation afin de résoudre des problèmes liés à la mémoire RAM et de permettre la généralisation du modèle à l'utilisation d'images capturées à partir d'un appareil photo utilisant une trame de Bayer similaire à l'appareil photo Sony utilisé dans le papier original.

## Ressources nécessaires
- 64 GB RAM
- GTX 1080
- PyTorch >= 0.4.0 (1.0.0 et 1.6.0  ont aussi été testés)
- RawPy >= 0.10 (0.15 a aussi été testé)

Le programme a été testé sous Ubuntu 18.04, 16.04, et Windows 10.

## Téléchargement du dataset
Télécharger le jeu de données suivant les instructions du [code original](https://github.com/cchen156/Learning-to-See-in-the-Dark) et le dézipper dans un dossier nommé `dataset`.

```
pytorch-Learning-to-See-in-the-Dark
  ├── dataset
  │   ├── Sony
  │   │   ├── long
  |   │   ├── short
  .   .   .
```

## Entraînement
`python train_Sony.py --loss 'L1'/'L2'/'ssim' [Fonction de loss] --epoch (int) [Nombre d'epochs] --plot_loss True/False [Tracé de la loss ?] --save_freq (int) [Fréquence de sauvegarde des poids .pth] --val_freq [Fréquence de test sur le jeu de validation lors de l'entraînement] --black_level (int) [Niveau de noir]`
- Sauvegarde du modèle et génération du résultat toutes les 100 epochs par défaut. 
- Les modèles entraînés sont sauvegardés dans `saved_model/` et les images générées sont stockés dans `result_Sony/`.

## Test
### Téléchargement du modèle préentraîné
Vous pouvez télécharger le modèle préentraîné Pytorch [ici](https://drive.google.com/file/d/1qVYtDEObRAox8SDH4Tbqs2s117v7tFWG/view?usp=sharing) et le mettre dans le dossier `saved_model/`. <br/>

`python test_Sony.py --generalization True/False [Généralisation à d'autres images] --file [Traiter une image en particulier] --black_level (int) [Niveau de noir]`

- Le modèle préentraîné a subit un entraînement de 4000 epochs sur les images `.ARW` prises par l'appareil photo Sony mais il peut être utilisé pour inférer sur des images autres capturées par un appareil photo partageant la même trame de Bayer.
- L'étape de test prend seulement des images 1024 x 1024 du jeu de test. 
- Les résultats seront sauvegardés dans un fichier `test_result_Sony` avec `gt` pour les images ground truth, `ori` pour les images données en entrée, et `out` pour les images obtenues en sortie de l'algorithme.

### Licence
MIT License.
