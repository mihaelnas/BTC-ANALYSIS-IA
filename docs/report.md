# Rapport Méthodologique — LOB Predictor BTCUSDT

## 1. Introduction

Ce rapport documente la méthodologie employée pour évaluer le pouvoir prédictif de la
microstructure du carnet d'ordres (Limit Order Book, LOB) dans le cadre de la prédiction
à court terme du mouvement de prix du BTCUSDT.

### 1.1 Problématique

Le carnet d'ordres contient de l'information sur l'offre et la demande immédiates qui n'est
pas entièrement reflétée dans le prix courant. La question centrale est :

> *Les caractéristiques extraites du carnet d'ordres permettent-elles de prédire, de façon
> probabiliste et statistiquement significative, la direction du prix à très court terme ?*

### 1.2 Cadre et limites

Ce projet est strictement académique. Les coûts de transaction, la latence d'exécution et
le slippage ne sont pas modélisés. Les résultats ne constituent pas des recommandations
de trading.

---

## 2. Données

### 2.1 Source et collecte

- **Source** : Binance WebSocket Depth Stream
- **Paire** : BTCUSDT (Spot)
- **Fréquence** : 100ms (diff. depth updates)
- **Profondeur** : 20 niveaux bid/ask par snapshot
- **Format** : Apache Parquet avec compression zstd

### 2.2 Pipeline de collecte

Architecture producteur-consommateur asynchrone :

1. **Producteur** : connexion WebSocket, réception des mises à jour incrémentielles
2. **Reconstruction** : maintien d'un carnet d'ordres local (SortedDict, O(log n))
3. **Consommateur** : extraction de snapshots aplatis, écriture batch en Parquet

### 2.3 Qualité des données

- Détection automatique des gaps de séquence (`firstUpdateId` / `lastUpdateId`)
- Détection des carnets croisés (best_bid ≥ best_ask)
- Re-synchronisation automatique via REST API en cas d'anomalie

---

## 3. Feature Engineering

### 3.1 Features de microstructure

| Feature | Formule | Intuition |
|:---|:---|:---|
| Mid-price | (best_bid + best_ask) / 2 | Prix de référence |
| Spread | best_ask - best_bid | Coût implicite |
| Relative spread | spread / mid × 10000 bps | Spread normalisé |
| Microprice | (P_ask × V_bid + P_bid × V_ask) / (V_bid + V_ask) | Prix "juste" ajusté |
| OBI(k) | (V_bid_k - V_ask_k) / (V_bid_k + V_ask_k) | Déséquilibre au niveau k |
| OBI pondéré | Σ (1/k × OBI_k) / Σ (1/k) | Déséquilibre agrégé |
| OFI(k) | Δ(V_bid_k) - Δ(V_ask_k) | Flux d'ordres temporel |
| Profondeur cum. | Σ V_bid/ask_{1..20} | Liquidité disponible |
| Book pressure | depth_bid / depth_total | Pression relative |
| VWAP bid/ask | Σ(P×V) / Σ(V) par côté | Prix moyen pondéré |
| Log return(k) | ln(mid_t / mid_{t-k}) | Momentum |
| Spread vol. | σ_rolling(spread) | Volatilité micro |

### 3.2 Labeling

**Horizon** : 50 ticks = 5 secondes

**Classification** :
- UP : future_return > θ
- DOWN : future_return < −θ
- NEUTRAL : |future_return| ≤ θ

**Seuil adaptatif** : θ calibré sur la distribution empirique pour obtenir
un ratio de classes NEUTRAL ≈ 40%.

### 3.3 Normalisation

Rolling z-score avec fenêtre glissante de 5000 observations :
```
z_t = (x_t - μ_{t-w:t}) / σ_{t-w:t}
```
Strictement causal : seules les données passées sont utilisées.

---

## 4. Modélisation

### 4.1 Baselines

| Modèle | Description | Log-loss attendu |
|:---|:---|:---|
| Random | P(k) = 1/3 pour tout k | 1.0986 (= ln 3) |
| Prior | P(k) = distribution marginale | ~1.05–1.08 |
| Momentum | Continuation du dernier mouvement | Variable |
| OBI-threshold | Seuillage direct sur OBI | Variable |

### 4.2 LightGBM

- **Objectif** : multiclass softmax
- **Hyperparamètres** : optimisés via Optuna (50 trials, walk-forward CV)
- **Régularisation** : L1/L2, min_child_samples, subsample, colsample

### 4.3 Deep Learning (optionnel)

Trois architectures PyTorch :
- **CNN** : convolutions 1D sur les niveaux du carnet
- **LSTM** : séquences temporelles de features
- **CNN-LSTM** : extraction spatiale + modélisation temporelle

Entraînement avec early stopping, gradient clipping, et LR scheduling.

---

## 5. Évaluation

### 5.1 Walk-Forward Validation

Validation temporelle stricte — aucun shuffle :
- **Train** : fenêtre de 24h
- **Test** : fenêtre de 4h
- **Pas** : avancement de 4h

### 5.2 Métriques

| Métrique | Type | Description |
|:---|:---|:---|
| Log-loss | Probabiliste | Qualité des probabilités prédites |
| Brier Score | Probabiliste | Calibration multi-classe |
| Accuracy | Classification | Taux de prédictions correctes |
| F1-score (macro) | Classification | Performance équilibrée entre classes |
| Confusion matrix | Classification | Analyse des erreurs par classe |
| Reliability diagram | Calibration | Vérification de la calibration |

### 5.3 Tests statistiques

Comparaison des modèles via tests appariés sur les folds walk-forward.

---

## 6. Monitoring

### 6.1 Détection de drift

- **Test de Kolmogorov-Smirnov** : comparaison des distributions par feature
- **Population Stability Index (PSI)** :
  - PSI < 0.1 : pas de drift
  - 0.1 ≤ PSI < 0.25 : drift modéré
  - PSI ≥ 0.25 : drift significatif

### 6.2 Suivi de performance

- Log-loss et accuracy glissants
- Alerte si performance < baseline random
- Historique des métriques au fil du temps

---

## 7. Résultats

*Section à compléter après entraînement et évaluation.*

---

## 8. Discussion et perspectives

### Résultats attendus
- Le LOB contient un signal prédictif faible mais statistiquement significatif
- LightGBM devrait surpasser les baselines en log-loss
- Le signal se dégrade rapidement avec l'horizon de prédiction

### Perspectives
- Extension à d'autres paires (ETHUSDT, SOLUSDT)
- Incorporation de données de trades (tape)
- Architectures Transformer pour le LOB
- Étude de la profitabilité après coûts de transaction

---

## Références

1. Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks
   for Limit Order Books*. IEEE Transactions on Signal Processing.
2. Briola, A. et al. (2024). *Deep Limit Order Book Forecasting*. LOBFrame.
3. Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*.
   Journal of Financial Econometrics.
4. Sirignano, J. (2019). *Deep Learning for Limit Order Books*. Quantitative Finance.
