# Model Card — LOB Predictor BTCUSDT

## Model Details

- **Model Name**: LOB Predictor v1
- **Model Type**: LightGBM (Gradient Boosting Decision Tree), multi-class softmax
- **Task**: 3-class classification (DOWN / NEUTRAL / UP)
- **Prediction Horizon**: 5 seconds (50 ticks × 100ms)
- **Training Data**: BTCUSDT order book snapshots from Binance
- **Framework**: LightGBM 4.x via scikit-learn API
- **Date**: 2026

## Intended Use

### Primary Use
Évaluation académique du pouvoir prédictif de la microstructure du carnet d'ordres.

### Out-of-Scope Uses
- ❌ Trading automatisé en production
- ❌ Gestion de portefeuille
- ❌ Prédiction sur d'autres paires ou marchés
- ❌ Horizons de prédiction différents de 5 secondes

## Training Data

### Source
- **Exchange**: Binance (Spot)
- **Paire**: BTCUSDT
- **Flux**: WebSocket Depth Stream @ 100ms
- **Niveaux**: 20 niveaux bid/ask par snapshot

### Volume
- ~864 000 snapshots/jour
- ~500 Mo/jour en Parquet compressé (zstd)

### Preprocessing
1. Reconstruction du carnet d'ordres local depuis le flux WebSocket
2. Extraction de features de microstructure (OBI, OFI, microprice, etc.)
3. Labeling adaptatif avec seuil calibré pour ~30/40/30 (DOWN/NEUTRAL/UP)
4. Normalisation rolling z-score (fenêtre glissante, pas de leakage)

## Features

| Catégorie | Features |
|:---|:---|
| Imbalance | OBI aux niveaux 1, 3, 5, 10, 20 + OBI pondéré |
| Flow | OFI aux niveaux 1-5 + OFI agrégé |
| Price | Microprice, spread, spread relatif, déviation microprice |
| Depth | Profondeur cumulée bid/ask, pression du carnet |
| VWAP | Prix moyen pondéré bid/ask |
| Returns | Log returns à 1, 5, 10, 50 ticks |
| Volatility | Volatilité du spread (rolling 100) |

## Evaluation

### Methodology
- **Walk-forward validation** (train expanding window, test sliding window)
- Pas de shuffle aléatoire — respect strict de l'ordre temporel
- Train: 24h, Test: 4h, Step: 4h

### Metrics
| Métrique | Description |
|:---|:---|
| Log-loss | Qualité probabiliste (plus bas = mieux) |
| Brier Score | Calibration multi-classe |
| Accuracy | Taux de classification correcte |
| F1-score (macro) | Performance équilibrée entre classes |

### Baselines
| Baseline | Description | Log-loss attendu |
|:---|:---|:---|
| Random | Probabilités uniformes (1/3) | ~1.0986 (ln 3) |
| Prior | Distribution marginale des classes | ~1.05-1.08 |
| Momentum | Continuation du dernier mouvement | Variable |
| OBI seuil | Seuillage direct sur l'imbalance | Variable |

## Limitations and Biases

### Limites connues
1. **Pas de modélisation des coûts** : slippage, commissions, latence non pris en compte
2. **Paire unique** : entraîné et évalué uniquement sur BTCUSDT
3. **Régime-dépendant** : performance susceptible de varier selon les conditions de marché
4. **Horizon fixe** : optimisé pour 5 secondes uniquement

### Biais potentiels
- **Biais temporel** : les données d'entraînement couvrent une période spécifique
- **Biais de survie** : BTCUSDT est le marché le plus liquide, les résultats ne généralisent pas
- **Biais de sélection** : les features ont été choisies sur la base de la littérature existante

## Ethical Considerations

Ce modèle est conçu pour la recherche académique. Il ne doit pas être utilisé comme base
pour des décisions financières. Les marchés financiers sont complexes et un modèle de ce type
ne capture qu'une fraction infime de l'information disponible.

## Technical Specifications

- **Python**: 3.11+
- **LightGBM**: 4.x
- **PyTorch**: 2.x (pour les architectures DL alternatives)
- **Stockage**: Apache Parquet avec compression zstd
- **Déploiement**: Gradio sur Hugging Face Spaces
