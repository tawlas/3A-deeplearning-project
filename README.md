# Pipeline pour l'ensemble des données

## Entrainement et évaluation

### Génération de dataset

Emplacement: racine

Pour génerer le dataset, executer:

```
python generate_2d_dataset.py -n 1000 -o 2dDeepSDF/data/random_obstacle/BW
```

où l'argument -n est le nombre d'images dans le dataset.

### Zones libres et zones d'obstacles

Emplacement: racine

Étant donné une image, ce script renvoie les coordonnées de l'espace libre et les coordonnées des obstacles. Enregistre 2 fichiers json (zones libres et zones d'obstacles) où les clés sont les noms des images et les valeurs la liste des coordonnées.

```
python get_zones.py -i 2dDeepSDF/data/random_obstacle/BW -o 2dDeepSDF/data/random_obstacle
```

### Début et objectif

Emplacement: racine

Ce script échantillonne les points de départ et de but d'une trajectoire. Dans l'espace libre pour les trajectoires sans collision et dans la zone d'obstacles pour les trajectoires avec collision.

1. Dans la zone d'obstacles.
   Ce cas est pour générer une trajectoire qui heurtera des obstacles.

```
python start_goal_obs.py -c 2dDeepSDF/data/random_obstacle/obstacle_zone.json -o 2dDeepSDF/data/random_obstacle -n 100
```

2. Dans l'espace libre
   Ce cas est pour générer une trajectoire qui évitera les obstacles.

```
python start_goal.py -c 2dDeepSDF/data/random_obstacle/free_zone.json -o 2dDeepSDF/data/random_obstacle -n 100
```

### Deep sdf

Emplacement: 2dDeepSDF

1. Prétraiter les images sur la carte SDF
   Ce script crée la carte sdf associée aux images (environnements).

```
python generateSDF2D.py -i data/random_obstacle/BW -o data/random_obstacle/SDF/train
```

2. Entrainement du 2d DeepSDF

```
python train_deep_sdf.py -e chomp256
```

3. Inférer les latents vectors.
   Le script suivant déduit le latent vector et calcule le sdf prédit correspondant à partir de la carte sdf. Les deux sont enregistrés dans le dossier "Reconstructions".
   Les entrées sont le fichier npy pour chaque carte sdf (différent du modèle d'entraînement où il s'agissait d'un seul npy pour tous les fichiers sdf).
   La source de données doit donc contenir un groupe de fichiers sdf. Le script génère un fichier sdf npy pour le latent vector et un autre fichier npy qui est la carte sdf prédite (regardez le modèle pour comprendre le fonctionnement de deepsdf).

```
python infer_sdf.py -e chomp256 -c 400
```

4. Test

- Générer des cartes SDF

```
python generateSDF2D.py -i data/test/test_images -o data/test/test_sdf
```

- Inférer les latents vectors et les sdf prédits
  Définir les chemins d'entrée et de sortie dans specs.json, puis exécuter:

```
python infer_sdf.py -e chomp256 -c 400
```

### Génération de trajectoires avec RRTStar

Emplacement: RRTStar

Enregistre un fichier JSON pour chaque image où la clé est le numéro de la trajectoire et la valeur est la trajectoire (liste de points).

1. Trajectoires sans collision:

```
python rrt_star.py --sg_path ../2dDeepSDF/data/random_obstacle/start_goal.json -c ../2dDeepSDF/data/random_obstacle/free_zone.json -o ../autoencoder/data/trajectories
```

2. Trajectoires avec collision:
   L'idée est de générer un chemin dans la zone d'obstacles. Puis extrapoler (linéairement par défaut)

```
python rrt_star.py --sg_path ../2dDeepSDF/data/random_obstacle/start_goal_obs.json -c ../2dDeepSDF/data/random_obstacle/obstacle_zone.json -o ../autoencoder/data/trajectories_obs
```

### Interpolation de trajectoire

Emplacement: Racine

Interpolation pour les trajectoires sans collision.
Interpole les trajectoires générées par RRTStar pour que toutes les trajectoires aient le même nombre de points.
Enregistre un fichier JSON pour chaque image où la clé est le numéro de la trajectoire et la valeur est la trajectoire.
Les trajectoires sont normalisées (diviser chaque coordonnée par la taille de l'image) de manière à avoir des valeurs comprises entre 0 et 1 pour une meilleure commodité lors de l'entraînement.

```
python interpolate.py -d autoencoder/data/trajectories -n 15 -o autoencoder/data/trajectories_interpolated
```

### Extrapolation (& intra) de trajectoire pour les zones d'obstacles.

Emplacement: Racine

Ce script extrapole les trajectoires générées à l'intérieur des zones d'obstacles. En même temps, il intrapole les trajectoires de manière à avoir le même nombre de points pour chacune d'elles.
Enregistre un fichier JSON pour chaque image où la clé est le numéro de la trajectoire et la valeur est la trajectoire.
Les trajectoires sont normalisées (divisées chaque coordonnée par la taille de l'image) de manière à avoir des valeurs comprises entre 0 et 1 pour une meilleure commodité lors de l'entraînement.

```
python extrapolate.py -d autoencoder/data/trajectories_obs -o autoencoder/data/trajectories_obs_interpolated -n 64
```

### Auto encoder

Emplacement: autoencoder

1. Prétraitement des données:
   Cette étape prend toutes les trajectoires (sans collision) dans les différents fichiers JSON (un JSON par image) et les concatènent dans un ensemble de données au format numpy pour former l'ensemble de données d'entraînement (et de validation).
   Enregistre un seul fichier numpy contenant un certain nombre de trajectoires dans tous les environnements.

Reproduisez ceci pour des données réelles et fausses (changez les chemins ci-dessous).

- Trajectoires sans collision:

```
python preprocess_ae.py -t data/trajectories_interpolated -o data/trajectories.npy
```

- Trajectoires avec collision:

```
python preprocess_ae.py -t data/trajectories_obs_interpolated -o data/trajectories_obs.npy
```

2. Entrainement de l'autoencoder:
   Les paramètres / spécifications d'entraînement sont dans un fichier JSON à l'intérieur du répertoire du modèle.

```
python train_ae.py -e autoencoder
```

3. Inférence:
   Le script prend tous les fichiers JSON dans un dossier contenant chacun, un ensemble de trajectoires relatives à un environnement particulier, encode les trajectoires et les enregistre au format numpy pour chaque fichier JSON.

Reproduisez ceci pour des données réelles et fausses (changez les chemins dans specs.json).

```
python infer_ae.py -e autoencoder -c 950
```

### CGAN

1. Préparation des données:
   Concatène une trajectoire interpolée et son latent environment correspondant.
   Enregistre un fichier numpy de plusieurs concaténations pour chaque environnement.

- Trajectoires sans collision:

```
python preprocess_cgan.py -t ../autoencoder/data/trajectories_interpolated -e ../2dDeepSDF/chomp256/Reconstructions/Codes   -o data/real
```

- Trajectoires avec collision:

```
python preprocess_cgan.py -t ../autoencoder/autoencoder/Codes/fake -e ../2dDeepSDF/chomp256/Reconstructions/fake/Codes   -o data/fake -f
```

2. Entrainemant du discriminateur:

```
python train_cgan.py -e models

```

### Discriminateur

1. Préparez les données:
   Concatène une trajectoire latente, son environnement latent correspondant et son label (éviter ou heurter un obstacle).
   Enregistre un fichier numpy de nombreuses concaténations pour chaque environnement.
   Reproduisez ceci pour les trajectoires d'évitement ainsi que pour les trajectoires de collision (changez la sortie selon dossier «réel» ou «faux»).

- Trajectoires sans collision:

```
python preprocess_d.py -t ../autoencoder/autoencoder/Codes/real -e ../2dDeepSDF/chomp256/Reconstructions/Codes  -l 1 -o data/real
```

- Trajectoires avec collision:

```
python preprocess_d.py -t ../autoencoder/autoencoder/Codes/fake -e ../2dDeepSDF/chomp256/Reconstructions/fake/Codes -l 0  -o data/fake
```

2. Entrainement du discriminateur:

```
python train_discriminator.py -e discriminator
```

### Generateur

Prend un vecteur d'environnement et les points début et fin pour générer une trajectoire.

#### Fully Connected Layers Generator

Emplacement: generator

1. Préparation des données:
   Concatène une trajectoire interpolée et son environnement latent correspondant.
   Enregistre un fichier numpy de plusieurs concaténations pour chaque environnement.
   Fait uniquement pour les trajectoires réelles (sans collision).

```
python preprocess_g.py -t ../autoencoder/data/trajectories_interpolated -e ../2dDeepSDF/chomp256/Reconstructions/Codes   -o data
```

2. Entrainement du generator:

```
python train_generator.py -e generator
```

3. Inférence:
   Le script prend tous les fichiers JSON dans un dossier contenant chacun, un ensemble de trajectoires relatives à un environnement particulier, encode les trajectoires et les enregistre au format numpy pour chaque fichier JSON.

- Convertit l'image en fichier sdf npy
- Deepsdf le transforme en vecteur npy
- start_goal algo affiche un fichier JSON de début et d'objectifs
- infer_generator charge le vecteur npy et concatène le vecteur sdf et le début etc ... pour le deuxième point et ainsi de suite

  Entrée: le dossier de données d'environnement et le chemin start_goal dans specs.json
  Enregistre un fichier JSON dont les clés sont les noms des environnements et valorise l'ensemble des trajectoires générées dans un environnement donné.

```
python infer_generator.py -e generator -c latest
```

## Tests et resultats d'évaluations

### Obtenir les zones des environnements de test

Emplacement: Racine

TODO: J'ai corrigé le retour x et y d'origine, donc je dois également le corriger dans le script intrapolate et extrapolate

```
python get_zones.py -i metrics/test_images -o metrics -s test
```

### Obtenir les coordonnées de début et d'arrivée

Emplacement: Racine

```
python start_goal.py -c metrics/free_zone_test.json -o metrics -n 100
```

### CGAN

```
python test_cgan.py -e models
```

### CGAN sans trajectory loss

```
python test_cgan_wtl.py -e models
```

### Generator Fully Connected Single

```
python test_generator_fc_single.py -e generator
```

### RRT Star

```
python test_rrtstar.py -e ./
```
