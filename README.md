# RAG MultiDoc
Ce projet utilise le modèle de langage `meta-llama/Meta-Llama-3-8B-Instruct` via l'API HuggingFace pour analyser des documents PDF. Il crée un graphe de connaissances à partir des documents et permet de poser des questions basées sur ces documents. Le projet utilise des bibliothèques telles que `llama_index`, `pyvis`, et `IPython` pour construire et visualiser le graphe de connaissances. Ce projet est destiné aux utilisateurs souhaitant extraire et analyser des informations de plusieurs fichiers PDF.
   - [llama_index](https://docs.llamaindex.ai/en/stable/)
   - [API HuggingFace](https://huggingface.co/docs/api-inference/index)
   - [Pyvis](https://pyvis.readthedocs.io/en/latest/index.html)

Le code fourni implémente une version simplifiée de RAG (Recapitulative Augmented Generation) en utilisant une base de données vectorielle et un graphe de connaissances pour capturer les relations entre les entités dans des documents PDF.

Voici une explication de chaque composant et de son rôle dans cette implémentation :

1. **Configuration du modèle (Configuration du modèle de langage et des embeddings) :**
   - Le code configure un modèle de langage et un modèle d'embeddings en utilisant l'API HuggingFace. Le modèle de langage est utilisé pour générer des réponses basées sur le contexte des documents PDF, tandis que le modèle d'embeddings est utilisé pour extraire des représentations vectorielles des données textuelles.

2. **Sélection des fichiers PDF :**
   - L'utilisateur est invité à sélectionner les fichiers PDF à analyser. Ces fichiers seront utilisés pour construire le graphe de connaissances et générer des réponses aux requêtes de l'utilisateur.

3. **Saisie de la requête :**
   - L'utilisateur entre une requête textuelle qui sera utilisée pour interroger les documents PDF sélectionnés. Cette requête est utilisée pour générer une réponse pertinente basée sur le contenu des documents.

4. **Génération de la réponse :**
   - Le script génère une réponse basée sur les documents PDF et la requête de l'utilisateur. Il utilise le modèle de langage pour générer une réponse en fonction du contexte des documents et de la requête.

5. **Visualisation du graphe de connaissances :**
   - Le script crée un graphe de connaissances à partir des documents PDF. Ce graphe est utilisé pour capturer les relations entre les entités et les concepts dans les documents. Il peut être visualisé dans un fichier HTML pour une exploration interactive.

En résumé, ce code implémente une approche RAG en utilisant des techniques de traitement du langage naturel (NLP) pour générer des réponses contextuelles basées sur des documents PDF et des requêtes utilisateur. Le graphe de connaissances est utilisé pour capturer les relations entre les entités dans les documents, ce qui permet une compréhension plus approfondie du contenu et une génération de réponses plus précises.
 
   - **[Documentation](https://github.com/yacineberkani/test/blob/main/Documentation.md)**


## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine:

- Python 3.7 ou supérieur

## Installation des dépendances

Installez les dépendances nécessaires en utilisant pip:

```sh
pip install -r requirements.txt
```
## 

## Utilisation

1. **Configuration du modèle**:
   - Le modèle de langage et le modèle d'embeddings sont configurés en utilisant l'API HuggingFace avec un jeton API. Remplacez `HF_TOKEN` par votre propre jeton API HuggingFace.

2. **Sélection des fichiers PDF**:
   - Sélectionnez les fichiers PDF à analyser lorsque la fenêtre de dialogue s'ouvre.

3. **Saisie de la requête**:
   - Entrez votre requête lorsque vous y êtes invité.

## Fonctionnalités supplémentaires

- **Visualisation du graphe de connaissances**:
  - La fonction `visualize_knowledge_graph` permet de visualiser le graphe de connaissances et de le sauvegarder en fichier HTML pour une exploration interactive.
  ![knowledge_graph](https://github.com/yacineberkani/test/blob/main/Capture%20d%E2%80%99e%CC%81cran%202024-05-23%20a%CC%80%2002.10.35.png)

## Auteur

BERKANI Yacine

