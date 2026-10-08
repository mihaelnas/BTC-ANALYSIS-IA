
# Étape 1 : construction des dépendances
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Environnement virtuel copié tel quel dans l'image finale
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# On copie uniquement requirements.txt pour profiter du cache Docker
COPY requirements.txt .
RUN pip install -r requirements.txt


# Étape 2 : image d'exécution
FROM python:3.11-slim AS runtime

# Installation de libgomp1 besoin par LightGBM sur une image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# Utilisateur non-root
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser main.py ./

# Dossiers de données montée en volume
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models /app/data/monitoring \
    && chown -R appuser:appuser /app/data

USER appuser

# Port de l'application Gradio
EXPOSE 7860

ENTRYPOINT ["python", "main.py"]
CMD ["help"]