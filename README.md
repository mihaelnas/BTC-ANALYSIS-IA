# LOB Predictor — BTCUSDT Order Book Price Prediction

> **Prédiction probabiliste du mouvement de prix à court terme à partir de la microstructure du carnet d'ordres**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Ce projet évalue le pouvoir prédictif de la microstructure du carnet d'ordres (Limit Order Book) pour prédire la direction du prix à très court terme sur le marché BTCUSDT.

**Ce projet est à vocation académique** — il ne vise pas à produire un système de trading rentable.

## 🏗 Architecture

```
PythonIA/
├── config/settings.py          # Configuration centralisée (pydantic-settings)
├── src/
│   ├── data/                   # Collecte et stockage
│   │   ├── collector.py        # WebSocket depth stream (asyncio)
│   │   ├── order_book.py       # Gestion du carnet local
│   │   └── storage.py          # Écriture Parquet optimisée
│   ├── features/               # Feature engineering
│   │   ├── microstructure.py   # OBI, OFI, microprice, VWAP...
│   │   ├── labels.py           # Labeling UP/DOWN/NEUTRAL
│   │   └── pipeline.py         # Pipeline d'orchestration
│   ├── models/                 # Modélisation
│   │   ├── baseline.py         # Baselines (Random, Prior, Momentum, OBI)
│   │   ├── gradient_boosting.py # LightGBM + Optuna
│   │   ├── deep_learning.py    # CNN, LSTM, CNN-LSTM (PyTorch)
│   │   └── evaluation.py       # Walk-forward validation + métriques
│   ├── monitoring/             # Monitoring
│   │   ├── drift_detector.py   # KS test + PSI
│   │   └── performance_tracker.py
│   └── deployment/
│       └── app.py              # Application Gradio
├── scripts/
│   ├── collect_data.py         # Lancer la collecte
│   ├── train_model.py          # Entraîner le modèle
│   └── evaluate_model.py       # Évaluer le modèle
└── tests/                      # Tests unitaires
```

## 🚀 Installation

```bash
# Cloner le projet
cd PythonIA

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Installer en mode développement
pip install -e ".[dev]"
```

## 📡 Étape 1 — Collecte de données

```bash
# Collecter indéfiniment (Ctrl+C pour arrêter)
python scripts/collect_data.py

# Dry run (30 secondes de test)
python scripts/collect_data.py --dry-run

# Collecter pendant 1 heure
python scripts/collect_data.py --duration 3600

# Collecte avec logs détaillés
python scripts/collect_data.py --duration 60 -v
```

Les données sont stockées en format Parquet dans `data/raw/`.

## 🔧 Étape 2 — Entraînement

```bash
# Pipeline complet : features + entraînement + évaluation
python scripts/train_model.py

# Avec optimisation des hyperparamètres (Optuna)
python scripts/train_model.py --optimize --n-trials 50

# Validation uniquement (pas de sauvegarde)
python scripts/train_model.py --validate-only --sample-size 10000
```

### Options utiles

- Override des fenêtres temporelles (en heures) pour la validation walk-forward :

```bash
# Exemple : réduire les fenêtres pour travailler sur peu de données
.venv/bin/python scripts/train_model.py \
	--train-window-hours 1 \
	--test-window-hours 1 \
	--step-hours 1
```

- On peut aussi passer ces valeurs via des variables d'environnement (préfixe `MODEL_`) :

```bash
MODEL_TRAIN_WINDOW_HOURS=1 MODEL_TEST_WINDOW_HOURS=1 MODEL_STEP_HOURS=1 \
	.venv/bin/python scripts/train_model.py
```

### Comportement d'auto-adaptation

Si les données disponibles couvrent une durée plus courte que les fenêtres demandées, le pipeline **réduit automatiquement** les fenêtres (train/test/step) de façon proportionnelle puis itérative jusqu'à obtenir au moins un split walk-forward valide. Cela permet d'exécuter `python scripts/train_model.py` sans erreurs même sur de petits jeux de données de développement. Si l'auto-ajustement échoue, le script lèvera une erreur expliquant la durée des données et les fenêtres essayées.

Note : pour la reproduction expérimentale, préférez définir explicitement les fenêtres via les flags ou les variables d'environnement.

## 📊 Étape 3 — Évaluation

```bash
# Évaluation walk-forward complète
python scripts/evaluate_model.py --with-baselines --save-plots
```

## 🌐 Étape 4 — Démo

```bash
# Lancer l'application Gradio localement
python src/deployment/app.py
```

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_order_book.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v
```

## 📐 Features de microstructure

| Feature | Description |
|:---|:---|
| **OBI** (Order Book Imbalance) | Déséquilibre bid/ask à N niveaux |
| **OFI** (Order Flow Imbalance) | Flux temporel d'ordres |
| **Microprice** | Prix ajusté par les volumes |
| **Spread** | Écart bid-ask (absolu et relatif) |
| **Cumulative Depth** | Profondeur cumulée par côté |
| **Book Pressure** | Ratio de profondeur bid/total |
| **VWAP** | Prix moyen pondéré par volume |
| **Log Returns** | Rendements à différents horizons |
| **Spread Volatility** | Volatilité du spread (rolling) |

## 🎯 Méthodologie d'évaluation

- **Walk-forward validation** : train sur [0,t], test sur [t, t+Δ], pas de shuffle
- **Métriques probabilistes** : Log-loss, Brier Score multi-classe
- **Métriques de classification** : F1-score (macro/weighted), matrice de confusion
- **Calibration** : Diagrammes de fiabilité par classe

## ⚠️ Limites

- Ce n'est **pas** un système de trading
- Coûts de transaction, latence et slippage non modélisés
- Évalué sur BTCUSDT uniquement
- Risque de data drift en production

## 📚 Références

- Zhang et al. (2019) — *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*
- Briola et al. (2024) — *LOBFrame: Deep Limit Order Book Forecasting*
- Cont et al. (2014) — *The Price Impact of Order Book Events*

## 📄 Licence

MIT
