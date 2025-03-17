# Chatbot pour Information sur le Diabète

## Description du projet

Ce projet s'inscrit dans le cadre de la certification en AI : Computer Vision and Natural Language Processing offerte par le programme FORCE-N et a été réalisé en groupe par Abdoul Wahab Soumare, Papa Seydou Wane, Pape Malick Diop et Adama Sall sous la supervision du mentor M. Fall.

Selon l'Organisation Mondiale de la Santé (OMS), la prévalence mondiale du diabète a doublé depuis 1980 et continue de progresser. Un diabète non maîtrisé à temps peut avoir de graves conséquences sur la santé et engendrer des coûts financiers importants. 

L'objectif de ce projet est de développer un chatbot capable de fournir des informations sur le diabète, telles que la définition, les symptômes, les types et les remèdes. Le projet a été réalisé en utilisant des technologies variées, avec une recommandation pour la librairie Rasa. Cependant, en raison de contraintes liées à la version de Python, nous avons opté pour Streamlit pour le déploiement.

## Structure du projet

```
|-- data
|   |-- about_diabete.csv
|   |-- classes.pkl
|   |-- intents.json
|   |-- words.pkl
|-- models
|   |-- chatbot-model-about-diabete.h5
|-- notebooks
|   |-- data-collecting.ipynb
|   |-- model-training.ipynb
|-- app.py
|-- README.md
|-- .gitignore
|-- requirements.txt
```

## Installation

1. **Cloner le dépôt**
```bash
git clone https://github.com/wahabsoumare/Diabete-chatbot
cd chatbot-diabete
```

2. **Créer un environnement virtuel**
```bash
python -m venv env
```

3. **Activer l'environnement virtuel**
- Sur Windows :
```bash
.\env\Scripts\activate
```
- Sur macOS/Linux :
```bash
source env/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Lancer l'application Streamlit**
```bash
streamlit run app.py
```

## Collecte et Préparation des Données

- **Source de données** : Les données ont été collectées via le scraping du site web spécialisé sur le diabète : [diabete.qc.ca](https://www.diabete.qc.ca/le-diabete-en-questions/).
- **Méthodologie** : Les membres de l'équipe (Soumare, Wane, Diop et Sall) ont divisé les sections à scraper et fusionné les données dans un dataset unique structuré en deux colonnes : `questions` et `answers`.
- **Nettoyage et prétraitement** : Alignement des questions et réponses, suppression des doublons et nettoyage des données inutiles.

## Développement du Modèle

- **Approches explorées** :
  - Similarité cosinus entre la question utilisateur et les questions du dataset.
  - Utilisation d'un modèle pré-entraîné `sentence_transformers`.
  - Modèle séquentiel avec TensorFlow (proposé par Somare, adopté pour la suite).

- **Architecture du modèle** :
```python
model = Sequential([
    Dense(units = 128, input_shape = (len(train_x[0]),), activation = 'relu', kernel_regularizer = 'l2' name = 'first_layer'),
    Dropout(rate = 0.25, name = 'first_dropout'),

    Dense(units = 64, activation = 'relu', kernel_regularizer = 'l2', name = 'second_layer'),
    Dropout(rate = 0.5, name = 'second_dropout'),

    Dense(units = len(train_y[0]), activation = 'softmax', name = 'final_layer')
], name = 'Chatbot_model')
```

- **Résultats** :
```
Accuracy: 0.9260
Loss: 1.3361
Validation Accuracy: 0.5784
Validation Loss: 3.8233
```
- Le modèle a montré de bonnes performances en répondant correctement à la plupart des questions.

## Déploiement

- Initialement, l'équipe a tenté de déployer avec Rasa et Twilio, mais des problèmes de compatibilité avec Python 12 ont empêché l'intégration.
- Solution finale : Déploiement via Streamlit sur un serveur public.
- Lien d'accès : [Chatbot Diabète](https://wahab-diabete-chatbot.streamlit.app/)

## Membres de l'équipe
- Abdoul Wahab Soumare
- Papa Seydou Wane
- Pape Malick Diop
- Adama Sall

## Références
- [Site Web Diabète Québec](https://www.diabete.qc.ca/le-diabete-en-questions/)

