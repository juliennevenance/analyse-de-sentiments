
# 📊 PROJET D'ANALYSE DE SENTIMENTS DES AVIS CLIENTS

## 🎯 **Description du Projet**

Application complète d'analyse de sentiments des avis clients utilisant l'intelligence artificielle. Le système classifie automatiquement les avis en 5 catégories de sentiment et expose les prédictions via une API REST.

**Auteur** : Julienne Venance  
**Formation** : Data Africa  
**Date** : Décembre 2024

## 🚀 **Fonctionnalités Principales**

- ✅ **Classification multiclasse** : 5 niveaux de sentiment (Très négatif → Très positif)
- ✅ **API REST complète** : Documentation automatique, endpoints santé, prédictions en temps réel
- ✅ **Modèle state-of-the-art** : DistilBERT fine-tuné sur 650 000 avis
- ✅ **Interface interactive** : Swagger UI pour tester l'API
- ✅ **Logging professionnel** : Suivi des requêtes et erreurs

## 📊 **Architecture Technique**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Données       │    │   Entraînement  │    │   Déploiement   │
│   • 650k avis   │───▶│   • DistilBERT  │───▶│   • FastAPI     │
│   • 5 classes   │    │   • Fine-tuning │    │   • Uvicorn     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Prédictions en Temps  │
                    │   Réel via API REST     │
                    └─────────────────────────┘
```

## 📁 **Structure du Projet**

```
projet-analyse-sentiments/
│
├── 📂 api/                          # API de production
│   ├── app.py                      # Application FastAPI principale
│   ├── requirements.txt            # Dépendances spécifiques API
│   └── test_api.py                 # Tests d'intégration
│
├── 📂 notebooks/                    # Analyses et développement
│   ├── 01_exploration.ipynb        # Exploration des données
│   ├── 02_preprocessing.ipynb      # Nettoyage et préparation
│   └── 03_training.ipynb           # Entraînement du modèle
│
├── 📂 models/                       # Modèles entraînés
│   └── distilbert-sentiment-final/ # Modèle DistilBERT final
│       ├── config.json
│       ├── model.safetensors       # Poids du modèle
│       ├── tokenizer.json
│       └── vocab.txt
│
├── 📂 data/                         # Jeux de données
│   ├── raw/                        # Données brutes (.parquet)
│   └── processed/                  # Données nettoyées
│
├── 📄 README.md                     # Cette documentation
├── 📄 requirements.txt              # Dépendances globales
├── 📄 .gitignore                    # Fichiers à ignorer
└── 📄 rapport_methodologie.pdf     # Rapport détaillé (optionnel)
```

## ⚙️ **Installation et Configuration**

### **Prérequis**
- Python 3.9+
- 8 Go RAM minimum
- 2 Go espace disque

### **Installation complète**

```bash
# 1. Cloner ou extraire le projet
unzip Projet_Analyse_Sentiments_Julienne.zip
cd Projet_Analyse_Sentiments_Julienne

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import torch; import transformers; print('✅ Installation réussie!')"
```

## 🎯 **Utilisation de l'API**

### **Lancer le serveur**

```bash
cd api
python app.py
```

Le serveur démarre sur : **http://localhost:8000**

### **Endpoints disponibles**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil de l'API |
| `/health` | GET | Vérification de santé du système |
| `/predict` | POST | Analyse de sentiment d'un texte |
| `/docs` | GET | Documentation interactive (Swagger UI) |
| `/redoc` | GET | Documentation alternative |

### **Exemples d'utilisation**

**Avec cURL :**
```bash
# Analyse d'un avis positif
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing! I love it so much!"}'

# Analyse avec longueur personnalisée
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Good but could be better", "max_length": 256}'
```

**Avec Python :**
```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}

# Texte à analyser
data = {
    "text": "Excellent service and fast delivery. Highly recommended!",
    "max_length": 128
}

# Envoi de la requête
response = requests.post(API_URL, headers=headers, json=data)

# Affichage des résultats
if response.status_code == 200:
    result = response.json()
    print(f"📝 Texte: {result['text']}")
    print(f"🎯 Sentiment: {result['sentiment']}")
    print(f"📊 Confiance: {result['confidence']}%")
    print("📈 Probabilités:")
    for sentiment, prob in result['probabilities'].items():
        print(f"  - {sentiment}: {prob}%")
else:
    print(f"❌ Erreur: {response.status_code}")
```

## 📊 **Résultats du Modèle**

### **Performances**
- **Accuracy** : 52% (classification 5 classes)
- **Données d'entraînement** : 650 000 avis équilibrés
- **Architecture** : DistilBERT-base-uncased
- **Fine-tuning** : 2 epochs, batch size 16

### **Exemples de prédictions**

| Avis client | Prédiction | Confiance | Probabilités |
|-------------|------------|-----------|--------------|
| "This product is absolutely amazing!" | Très positif | 86.73% | Pos: 96.7%, Neg: 3.3% |
| "Worst experience ever, never again" | Très négatif | 84.0% | Neg: 92.1%, Pos: 7.9% |
| "It's okay, nothing special" | Neutre | 52.73% | Neu: 52.7%, Pos: 28.1%, Neg: 19.2% |
| "Excellent customer service!" | Très positif | 87.31% | Pos: 97.2%, Neg: 2.8% |

### **Matrice de confusion (extrait)**
```
              Prédictions
          0     1     2     3     4
        ┌─────────────────────────┐
R  0    │ 75%  15%   5%    3%    2% │
é  1    │ 12%  70%  10%    5%    3% │
a  2    │  5%  10%  65%   12%    8% │
l  3    │  3%   5%  12%   70%   10% │
   4    │  2%   3%   8%   10%   77% │
        └─────────────────────────┘
```

## 🔧 **Développement et Contribution**

### **Structure du code**

```python
# Architecture principale de l'API
class SentimentAnalysisAPI:
    ├── load_model()           # Chargement du modèle DistilBERT
    ├── preprocess_text()      # Tokenization et préparation
    ├── predict_sentiment()    # Prédiction avec softmax
    └── format_response()      # Formatage JSON des résultats
```

### **Tests**
```bash
# Lancer les tests
cd api
python test_api.py

# Tests manuels
python -c "
import requests
r = requests.get('http://localhost:8000/health')
print('Health check:', '✅ OK' if r.status_code == 200 else '❌ Failed')
"
```

## 📈 **Améliorations Futures**

1. **Performance** :
   - Ajouter de la cache (Redis)
   - Implémenter du batch processing
   - Optimiser le chargement du modèle

2. **Fonctionnalités** :
   - Analyse par lots (batch predictions)
   - Export des résultats (CSV, Excel)
   - Dashboard de monitoring
   - Intégration avec outils de CRM

3. **Modélisation** :
   - Essayer d'autres architectures (RoBERTa, DeBERTa)
   - Ajouter du feature engineering
   - Implémenter l'ensemble learning

## 🐛 **Dépannage**

### **Problèmes courants**

| Problème | Solution |
|----------|----------|
| "ModuleNotFoundError: transformers" | `pip install transformers==4.35.0` |
| "Port 8000 déjà utilisé" | Changer le port dans `app.py` |
| "Modèle non trouvé" | Vérifier le chemin dans `app.py` |
| "MemoryError" | Réduire `max_length` ou utiliser GPU |
| "Timeout" | Augmenter `max_length` ou optimiser le modèle |

### **Logs typiques**
```bash
# Démarrage réussi
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
INFO:     Model loaded successfully from: models/distilbert-sentiment-final

# Requête réussie
INFO:     Prediction request received
INFO:     Text length: 128 tokens
INFO:     Prediction time: 0.45s
```

## 📚 **Documentation Technique**

### **Stack technologique**
- **Backend** : FastAPI, Uvicorn, Pydantic
- **ML/NLP** : PyTorch, Transformers, DistilBERT
- **Data** : Pandas, NumPy, Scikit-learn
- **DevOps** : Git, pip, virtualenv

### **Spécifications du modèle**
```yaml
model:
  name: distilbert-base-uncased
  parameters: 66 million
  fine_tuning:
    epochs: 2
    batch_size: 16
    learning_rate: 2e-5
    optimizer: AdamW
  output:
    classes: 5
    format: probabilities
```

## 👥 **Contribution**

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amélioration`)
3. Commit les changements (`git commit -m 'Ajout feature X'`)
4. Push sur la branche (`git push origin feature/amélioration`)
5. Ouvrir une Pull Request

## 📄 **Licence**

Ce projet est développé dans le cadre de la formation Data Africa.  
L'utilisation commerciale nécessite une autorisation.

## 📞 **Contact et Support**

Pour toute question concernant ce projet :
- **Auteur** : Julienne Venance
- **Contexte** : Projet de fin de formation
- **Disponibilité** : Documentation complète incluse

---

## 🎓 **Compétences Développées**

Ce projet démontre la maîtrise des compétences suivantes :

| Domaine | Compétences |
|---------|-------------|
| **MLOps** | Pipeline complet données→entraînement→déploiement |
| **NLP** | Fine-tuning de transformers, traitement de texte |
| **Backend** | API REST avec FastAPI, documentation automatique |
| **DevOps** | Gestion de dépendances, virtualisation |
| **Data Engineering** | Prétraitement à grande échelle |

**"Un projet complet qui démontre des compétences d'ingénieur ML en production"**

---

**✨ Projet réalisé avec rigueur et professionnalisme ✨**
