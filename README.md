# REXIA Projet 2026

## Groupe 6

### Gaël GARNIER - Raphaël VIGNAL - Maëlys HANOIRE - Fanny BADOULES

Projet d'analyse de biais dans des modèles de machine learning sur deux types de données : données RH tabulaires et données images.

## Partie 1 - Données RH

Prédiction de la démission d'employés à partir d'un dataset RH.

* Analyse exploratoire et prétraitement des données
* Analyse des variables sensibles et biais potentiels
* Modèles : XGBoost, Régression Logistique, GAM
* Évaluation de l'équité (Disparate Impact)
* Explication post-hoc : SHAP
* Intervention : suppression des variables sensibles et des fuites de données

## Partie 2 - Données images

Classification binaire souriant / non souriant sur le dataset CelebA, avec analyse des biais de genre.

* Analyse exploratoire du dataset CelebA
* Fine-tuning d'un ResNet18 pré-entraîné
* Évaluation de l'équité : accuracy, FPR, FNR par genre
* Explication post-hoc : GradCAM et LIME
* Intervention : crop du visage (OpenCV CascadeClassifier) pour réduire les biais liés au fond

## Dépendances

Pour ajouter la librairie à votre projet :

```
uv add "la librairie"
```

```
uv venv --python 3.12
```
