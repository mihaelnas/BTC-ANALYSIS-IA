# LOB Predictor — BTCUSDT Order Book Price Prediction

> **Prédiction probabiliste du mouvement de prix à court terme à partir de la microstructure du carnet d'ordres**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

Ce projet évalue le pouvoir prédictif de la microstructure du carnet d'ordres (Limit Order Book) pour prédire la direction du prix à très court terme (horizon 5 secondes) sur le marché BTCUSDT.

**Ce projet est à vocation académique** — il ne vise pas à produire un système de trading rentable en production (les coûts de transaction et la latence ne sont pas modélisés).

---

## Guide d'utilisation de la CLI (main.py)

L'application dispose d'une interface en ligne de commande unifiée via `main.py` (ou `scripts/cli.py`).

### 1. Afficher l'aide interactive

```bash
# Guide général et liste des commandes
python main.py help

# Aide détaillée sur une commande spécifique (ex: collect, train, evaluate)
python main.py help collect
python main.py help train
```

### 2. Collecte des données du carnet d'ordres (`collect`)

Collecte les snapshots du carnet d'ordres BTCUSDT via le WebSocket Binance (100ms) et les sauvegarde en format Parquet dans `data/raw/`.

```bash
# Collecte standard (60 secondes par défaut)
python main.py collect

# Collecter pendant 5 minutes (300 secondes) avec 10 niveaux de profondeur
python main.py collect --duration 300 --n-levels 10

# Mode test sans sauvegarde (dry-run)
python main.py collect --dry-run --duration 30
```

### 3. Calcul des caractéristiques et étiquetage (`features`)

Traite les fichiers Parquet bruts pour calculer 28 caractéristiques de microstructure, les labels adaptatifs (`DOWN`, `NEUTRAL`, `UP`) et la normalisation rolling z-score.

```bash
# Générer le jeu de données pour l'entraînement (enregistré dans data/processed/features_labeled.parquet)
python main.py features
```

### 4. Entraînement du modèle LightGBM (`train`)

Entraîne le modèle de Gradient Boosting avec validation croisée temporelle (walk-forward).

```bash
# Entraînement standard
python main.py train

# Entraînement avec optimisation des hyperparamètres via Optuna (50 essais)
python main.py train --optimize --n-trials 50

# Test rapide sur un échantillon restreint de 15 000 lignes
python main.py train --sample-size 15000

# Validation sans sauvegarder l'artefact de modèle
python main.py train --validate-only
```

### 5. Évaluation et métriques (`evaluate`)

Évalue les performances du modèle LightGBM et les compare aux modèles baselines (Random, Prior, Momentum, OBI).

```bash
# Évaluation complète avec baselines et sauvegarde des graphiques
python main.py evaluate --with-baselines --save-plots

# Évaluation et mise à jour automatique du rapport de méthodologie docs/report.md
python main.py evaluate --with-baselines --generate-report
```

Les graphiques (matrice de confusion, calibration, walk-forward) sont enregistrés dans `data/models/plots/`.

### 6. Détection de Data Drift (`drift-check`)

Analyse la stabilité des variables entre une fenêtre de référence et la fenêtre actuelle via le test de Kolmogorov-Smirnov et le Population Stability Index (PSI).

```bash
# Détection de drift standard
python main.py drift-check

# Spécifier le dossier de sortie des rapports CSV
python main.py drift-check --output data/monitoring
```

Le rapport de synthèse est sauvegardé en format CSV dans `data/monitoring/drift_summary_*.csv`.

### 7. Lancement de l'application Web Gradio (`serve`)

Lance l'interface web interactive pour explorer le carnet d'ordres et effectuer des prédictions.

```bash
# Lancer localement sur http://localhost:7860
python main.py serve

# Lancer sur un port spécifique
python main.py serve --server-port 8000

# Générer un lien public partageable (Gradio share)
python main.py serve --share
```

---

## Installation

```bash
# Cloner le projet et se placer dans le répertoire
cd PythonIA

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Installer en mode développement
pip install -e ".[dev]"
```

---

## Architecture du projet

```text
PythonIA/
├── main.py                     # Point d'entrée principal de la CLI
├── config/settings.py          # Configuration centralisée (pydantic-settings)
├── src/
│   ├── data/                   # Collecte WebSocket Binance & stockage Parquet
│   ├── features/               # Feature engineering (OBI, OFI, microprice, labels)
│   ├── models/                 # Modèles (LightGBM, baselines, walk-forward CV)
│   ├── monitoring/             # Détection de drift (KS-test & PSI)
│   ├── deployment/             # Application Gradio (app.py)
│   └── utils/                  # UI CLI & formateurs Rich (cli_ui.py)
├── scripts/
│   ├── cli.py                  # Orchestrateur CLI
│   ├── collect_data.py         # Script de collecte
│   ├── train_model.py          # Script d'entraînement
│   └── evaluate_model.py       # Script d'évaluation
├── data/                       # Données brutes, traitées et modèles
├── docs/                       # Rapport de méthodologie & Model Card
└── tests/                      # Suite de tests unitaires pytest
```

---

## Exécution des Tests

```bash
# Lancer l'ensemble des tests unitaires
pytest

# Lancer des modules de test spécifiques
pytest tests/test_cli.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v
pytest tests/test_order_book.py -v
```

---

## Caractéristiques de Microstructure (Features)

| Feature | Description |
|:---|:---|
| **OBI** (Order Book Imbalance) | Déséquilibre bid/ask à N niveaux de profondeur |
| **OFI** (Order Flow Imbalance) | Flux temporel d'ordres d'achat/vente |
| **Microprice** | Prix ajusté par le volume relatif des meilleurs bid/ask |
| **Spread** | Écart bid-ask (absolu et relatif en bps) |
| **Cumulative Depth** | Profondeur cumulée bid/ask sur 20 niveaux |
| **Book Pressure** | Ratio de la profondeur bid sur la profondeur totale |
| **VWAP** | Prix moyen pondéré par les volumes (bid & ask) |
| **Log Returns** | Rendements logarithmiques (1, 5, 10, 50 ticks) |
| **Spread Volatility** | Volatilité glissante du spread |

---

## Limites

- Projets strictement académique (pas un système de trading automatisé).
- Les frais de courtage (maker/taker), la latence réseau et le slippage ne sont pas modélisés.
- Évalué spécifiquement sur BTCUSDT Spot.

---

## Licence

MIT
