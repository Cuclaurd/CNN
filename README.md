# Projet CNN

## Présentation

Ce projet constitue le début d'une implémentation d'un réseau de neurones
convolutif (CNN) en Python. La première étape consiste à charger et à
comprendre les images du dataset CIFAR-10 avant de construire le modèle de
classification.

Le programme utilise la version binaire du dataset CIFAR-10. Il lit les
images, sépare leurs labels, puis reconstruit chaque image sous la forme d'un
tableau NumPy RGB.

## Dataset CIFAR-10

Le dataset contient des images couleur de dimensions `32 x 32` réparties dans
10 classes :

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

Chaque image possède trois canaux de couleur :

```text
Rouge (R), Vert (G), Bleu (B)
```

Après chargement, une image a la forme :

```python
(32, 32, 3)
```

Les pixels ont des valeurs comprises entre `0` et `255`.

## Format binaire utilisé

Le projet utilise les fichiers binaires suivants :

```text
cifar-10-batches-bin/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
├── test_batch.bin
└── batches.meta.txt
```

Chaque fichier `.bin` contient 10 000 images. Une image est stockée sous la
forme :

```text
1 octet de label
1024 octets du canal rouge
1024 octets du canal vert
1024 octets du canal bleu
```

Une image occupe donc :

```text
1 + (32 x 32 x 3) = 3073 octets
```

Dans le fichier binaire, les canaux sont stockés séparément. Le programme les
reconstruit dans l'ordre utilisé par NumPy :

```text
(nombre_images, hauteur, largeur, canaux)
```

Pour un batch complet, la forme obtenue est :

```python
(10000, 32, 32, 3)
```

## Installation

Créer un environnement virtuel dans le dossier du projet :

```bash
python3 -m venv .venv
```

Activer l'environnement :

```bash
source .venv/bin/activate
```

Installer NumPy :

```bash
python -m pip install numpy
```

## Lancement

Depuis la racine du projet :

```bash
python CNN.py
```

Le programme charge la première image de `data_batch_1.bin`, affiche son
label, le nom de sa classe et les valeurs de ses pixels.

## Structure actuelle

```text
CNN/
├── CNN.py
├── README.md
└── cifar-10-batches-bin/
    ├── data_batch_1.bin
    └── batches.meta.txt
```

## Prochaines étapes

- afficher les images graphiquement ;
- préparer les données pour l'entraînement ;
- créer l'architecture du CNN ;
- entraîner le modèle sur les batches d'entraînement ;
- évaluer le modèle sur `test_batch.bin`.
