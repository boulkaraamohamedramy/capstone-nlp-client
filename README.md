# 🎓 Capstone IA — Classification de Messages Clients (NLP)

Projet Capstone du DEC en Intelligence Artificielle — La Cité College.  
Pipeline NLP complet pour la classification automatique de messages clients en **11 catégories**, avec fine-tuning d'un modèle Transformer multilingue **XLM-RoBERTa** (FR/EN).

> 🚧 *Projet en cours — Module 1 disponible. Modules suivants à venir (fin avril 2026).*

---

## 📌 Objectif

Construire un système de classification automatique des messages de service client capable de :
- Identifier le type de message parmi 11 catégories (ORDER, REFUND, ACCOUNT, DELIVERY, etc.)
- Fonctionner en **bilingue français et anglais**
- Surpasser une baseline classique TF-IDF grâce au fine-tuning Transformer

---

## 👥 Équipe

- **Boulkaraa Mohamed Ramy**
- **Aksil Abdelkhalek**

**Encadrant :** Stéphanie N. Kahindo

---

## 📊 Dataset

- **Source** : Dataset Bitext (messages clients multilingues)
- **Volume** : 26 872 messages · 11 catégories
- **Langues** : Français & Anglais
- **Défi** : Déséquilibre des classes (ACCOUNT ~22% vs CANCEL ~3%)

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![XLM-RoBERTa](https://img.shields.io/badge/XLM--RoBERTa-412991?style=flat-square&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🔍 Module 1 — Classification du Type de Message

### Pipeline complet
1. Chargement et EDA du dataset Bitext (26 872 messages)
2. Prétraitement et nettoyage des données
3. Détection automatique de la langue (fastText)
4. Baseline TF-IDF + Logistic Regression
5. Fine-tuning XLM-RoBERTa multilingue (279M paramètres)
6. Évaluation : F1 macro, accuracy, matrice de confusion
7. Démo d'inférence sur messages réels FR/EN

### Résultats

| Modèle | F1 Macro | Accuracy |
|--------|----------|----------|
| Baseline TF-IDF + LR | — | — |
| **XLM-RoBERTa Fine-Tuning ✅** | **~0.99** | **~0.99** |

> XLM-RoBERTa atteint une confiance de 99.7% à 99.99% sur les messages de test FR/EN.

### Choix techniques
- **XLM-RoBERTa** : pré-entraîné sur 100+ langues → nativement bilingue FR/EN
- **F1 macro** : métrique adaptée au déséquilibre des classes
- **Split stratifié 80/10/10** : préserve les proportions des classes minoritaires
- **max_length=128** : couvre 99%+ des messages (P95 < 50 mots)
- **EarlyStopping + warmup_ratio=0.1** : stabilise le fine-tuning

---

## 🗺️ Modules à venir

| Module | Description | Statut |
|--------|-------------|--------|
| Module 1 | Classification du type de message (11 classes) | ✅ Terminé |
| Module 2 | Classification du niveau d'urgence (4 classes) | 🔄 En cours |
| Module 3 | À venir | ⏳ Prévu |

---

## 📁 Structure du repo

```
capstone-nlp-client/
│
├── Module1_Classification_Type_Message.ipynb   # Pipeline complet Module 1
└── README.md
```

---

## 👤 Auteur

**Mohamed Ramy Boulkaraa**  
Étudiant en Intelligence Artificielle — La Cité, Ottawa  
[GitHub](https://github.com/boulkaraamohamedramy)
