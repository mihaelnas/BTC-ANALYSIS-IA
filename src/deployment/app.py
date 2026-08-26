"""
Gradio application for LOB Predictor demo.

Deployable on Hugging Face Spaces.
Provides:
- Upload historical data or connect live
- Feature visualization
- Real-time prediction display with probability gauge
- Model performance dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def load_model():
    """Load the pre-trained LightGBM model."""
    from config.settings import MODELS_DIR
    model_path = MODELS_DIR / "lgbm_model.txt"

    if not model_path.exists():
        return None

    from src.models.gradient_boosting import LightGBMModel
    model = LightGBMModel()
    model.load(model_path)
    return model


def predict_from_csv(file_obj) -> tuple[str, plt.Figure | None, str]:
    """
    Process uploaded CSV/Parquet file and generate predictions.

    Returns:
        Tuple of (predictions text, plot, stats)
    """
    model = load_model()
    if model is None:
        return "No trained model found. Please train a model first.", None, ""

    # Load data
    if file_obj is None:
        return "Please upload a file.", None, ""

    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Compute features if raw data
    if "obi_1" not in df.columns:
        from src.features.microstructure import compute_all_features
        features = compute_all_features(df)
    else:
        features = df

    # Prepare X
    exclude_cols = {"timestamp", "mid_price", "microprice", "label", "future_return"}
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    X = features[feature_cols].dropna()

    if X.empty:
        return "No valid samples after preprocessing.", None, ""

    # Predict
    proba = model.predict_proba(X)
    preds = np.argmax(proba, axis=1)

    label_names = {0: "DOWN ↓", 1: "NEUTRAL →", 2: "UP ↑"}

    # Summary
    n = len(preds)
    counts = {label_names[i]: int((preds == i).sum()) for i in range(3)}
    avg_confidence = float(np.max(proba, axis=1).mean())

    summary = f"**{n} predictions generated**\n\n"
    for label, count in counts.items():
        pct = count / n * 100
        summary += f"- {label}: {count} ({pct:.1f}%)\n"
    summary += f"\n**Avg confidence:** {avg_confidence:.2%}"

    # Plot probability distribution over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Stacked probability plot
    ax = axes[0]
    x_range = range(min(len(proba), 500))  # Show last 500 predictions
    proba_subset = proba[-500:]
    ax.fill_between(x_range, 0, proba_subset[:, 2], alpha=0.6, color="#2ecc71", label="UP")
    ax.fill_between(x_range, proba_subset[:, 2], proba_subset[:, 2] + proba_subset[:, 1],
                     alpha=0.6, color="#95a5a6", label="NEUTRAL")
    ax.fill_between(x_range, proba_subset[:, 2] + proba_subset[:, 1], 1,
                     alpha=0.6, color="#e74c3c", label="DOWN")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Probabilities Over Time")
    ax.legend(loc="upper right")
    ax.set_xlim([0, len(x_range)])
    ax.set_ylim([0, 1])

    # Prediction histogram
    ax = axes[1]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    ax.bar(list(counts.keys()), list(counts.values()), color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")

    plt.tight_layout()

    # Stats
    stats_text = json.dumps({
        "total_predictions": n,
        "avg_confidence": round(avg_confidence, 4),
        **{k: v for k, v in counts.items()},
    }, indent=2)

    return summary, fig, stats_text


def create_demo_predictions() -> tuple[str, plt.Figure, str]:
    """Generate demo predictions with synthetic data."""
    np.random.seed(42)
    n = 200

    # Simulate probability outputs
    proba = np.random.dirichlet([2, 3, 2], size=n)
    preds = np.argmax(proba, axis=1)

    label_names = {0: "DOWN ↓", 1: "NEUTRAL →", 2: "UP ↑"}
    counts = {label_names[i]: int((preds == i).sum()) for i in range(3)}

    summary = f"**Demo: {n} synthetic predictions**\n\n"
    for label, count in counts.items():
        pct = count / n * 100
        summary += f"- {label}: {count} ({pct:.1f}%)\n"

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(n)
    ax.fill_between(x, 0, proba[:, 2], alpha=0.6, color="#2ecc71", label="UP")
    ax.fill_between(x, proba[:, 2], proba[:, 2] + proba[:, 1],
                     alpha=0.6, color="#95a5a6", label="NEUTRAL")
    ax.fill_between(x, proba[:, 2] + proba[:, 1], 1,
                     alpha=0.6, color="#e74c3c", label="DOWN")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time step")
    ax.set_title("Demo: Predicted Probabilities")
    ax.legend()
    plt.tight_layout()

    return summary, fig, json.dumps(counts, indent=2)


def build_app() -> gr.Blocks:
    """Build the Gradio application."""
    with gr.Blocks(
        title="LOB Predictor — BTCUSDT",
    ) as app:
        gr.Markdown(
            """
            # LOB Predictor — BTCUSDT Order Book

            **Prédiction probabiliste du mouvement de prix à court terme
            à partir de la microstructure du carnet d'ordres**

            Ce modèle estime la probabilité d'un mouvement haussier (UP),
            baissier (DOWN) ou neutre (NEUTRAL) dans les 5 prochaines secondes,
            à partir de snapshots du carnet d'ordres.

            ---
            """
        )

        with gr.Tabs():
            with gr.Tab("Prédiction"):
                gr.Markdown("### Téléchargez un fichier de données ou lancez la démo")

                with gr.Row():
                    file_input = gr.File(
                        label="Données order book (CSV ou Parquet)",
                        file_types=[".csv", ".parquet"],
                    )
                    demo_btn = gr.Button("Lancer la démo", variant="secondary")

                predict_btn = gr.Button("Prédire", variant="primary")

                with gr.Row():
                    summary_output = gr.Markdown(label="Résumé")
                    stats_output = gr.Code(label="Statistiques (JSON)", language="json")

                plot_output = gr.Plot(label="Visualisation des probabilités")

                predict_btn.click(
                    fn=predict_from_csv,
                    inputs=[file_input],
                    outputs=[summary_output, plot_output, stats_output],
                )

                demo_btn.click(
                    fn=create_demo_predictions,
                    outputs=[summary_output, plot_output, stats_output],
                )

            with gr.Tab("Model Card"):
                gr.Markdown(
                    """
                    ## Model Card

                    ### Description
                    - **Modèle** : LightGBM (gradient boosting, multi-class softmax)
                    - **Entrée** : Features de microstructure extraites de 20 niveaux du carnet d'ordres
                    - **Sortie** : Probabilités pour 3 classes (DOWN, NEUTRAL, UP)
                    - **Horizon** : 5 secondes (50 ticks à 100ms)

                    ### Features utilisées
                    - Order Book Imbalance (OBI) à 5 niveaux
                    - Order Flow Imbalance (OFI)
                    - Microprice et spread
                    - Profondeur cumulée et pression du carnet
                    - VWAP bid/ask
                    - Rendements logarithmiques (1, 5, 10, 50 ticks)
                    - Volatilité du spread

                    ### Limites
                    - Ce modèle n'est PAS un système de trading
                    - Les coûts de transaction, la latence et le slippage ne sont pas modélisés
                    - Performance évaluée sur BTCUSDT uniquement (pas de garantie de généralisation)
                    - Le modèle peut subir une dégradation de performance en cas de changement
                      de régime de marché (drift)

                    ### Évaluation
                    - Validation temporelle stricte (walk-forward, sans shuffle)
                    - Métriques : log-loss, Brier score, F1-score macro
                    """
                )

            with gr.Tab("ℹ️ À propos"):
                gr.Markdown(
                    """
                    ## À propos du projet

                    Ce projet évalue le pouvoir prédictif de la microstructure du carnet d'ordres
                    (order book) dans un cadre académique rigoureux.

                    ### Données
                    - Source : Binance WebSocket (depth stream 100ms)
                    - Paire : BTCUSDT
                    - Format : Snapshots de 20 niveaux bid/ask

                    ### Stack technique
                    - Python 3.11+, LightGBM, PyTorch
                    - Feature engineering : pandas, numpy
                    - Déploiement : Gradio, Hugging Face Spaces

                    ### Références
                    - Zhang et al. (2019) — DeepLOB
                    - Briola et al. (2024) — LOBFrame
                    - Cont et al. (2014) — Order Flow Imbalance
                    """
                )

    return app


if __name__ == "__main__":
    app = build_app()
    # Gradio 6 moved some parameters from Blocks(...) to launch()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
