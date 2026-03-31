# REXIA Projet 2026

## Groupe 6 - fichier final : Gr06_REXIA2026.ipynb

### Gaël GARNIER - Raphaël VIGNAL - Maëlys HANOIRE - Fanny BADOULES

Projet d'analyse de biais dans des modèles de machine learning sur deux types de données : données tabulaires, données images et données textuelles.

## Partie 1 - Données tabulaires

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

## Partie 3 - Données textuelles

- Dataset Civil Comments, 30 000 commentaires avec labels de toxicité et annotations démographiques (genre, couleur de peau, orientation sexuelle)
- Analyse exploratoire : longueurs des textes, distribution des classes, équilibre du dataset
- Prétraitement : nettoyage, tokenisation (NLTK), lemmatisation, nuages de mots toxiques / non-toxiques
- Visualisation TF-IDF par classe pour identifier les termes discriminants
- Analyse démographique : taux de toxicité par groupe sensible (female, male, black, white, homosexual)
- Classification Régression Logistique (80/20) avec évaluation et commentaires
- Explicabilité : SHAP (summary plot, waterfall, PDP) et LIME pour expliquer les décisions du modèle
- Fairness : analyse à deux niveaux dataset (taux de toxicité par groupe) et modèle (FPR, FNR, selection rate par groupe) via Fairlearn

## Dépendances

Pour ajouter la librairie à votre projet :

```
uv add "la librairie"
```

```
uv venv --python 3.12
```
