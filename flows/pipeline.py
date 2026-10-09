from prefect import flow, task
import subprocess
import sys
import platform

@task(name="Collecte des données")
def run_collect(duration: int = 60):
    """Lance la collecte du carnet d'ordres"""
    if platform.system() == "Windows":
        print("Attention : la collecte WebSocket a des limitations sur Windows.")
        print("On passe cette étape pour le test de l'orchestration.")
        return "Collecte ignorée (Windows)"

    result = subprocess.run(
        [sys.executable, "main.py", "collect", "--duration", str(duration)],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Échec de la collecte")
    return "Collecte terminée"


@task(name="Calcul des features")
def run_features():
    """Calcule les 28 features + labels"""
    result = subprocess.run(
        [sys.executable, "main.py", "features"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("Note : features a échoué (peut-être pas de données brutes).")
        return "Features non exécutées"
    return "Features terminées"


@task(name="Entraînement du modèle")
def run_train():
    """Entraîne le modèle LightGBM"""
    result = subprocess.run(
        [sys.executable, "main.py", "train"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return "Entraînement non exécuté"
    return "Entraînement terminé"


@task(name="Vérification du drift")
def run_drift_check():
    """Vérifie s'il y a du data drift"""
    result = subprocess.run(
        [sys.executable, "main.py", "drift-check"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return "Drift check terminé"


@flow(name="Pipeline BTC LOB", log_prints=True)
def btc_pipeline(duration: int = 60, do_train: bool = False):
    """
    Pipeline d'orchestration Prefect
    """
    print("=== Démarrage du pipeline BTC-ANALYSIS-IA (Orchestration) ===")

    collect_result = run_collect(duration)
    print(collect_result)

    features_result = run_features()
    print(features_result)

    drift_result = run_drift_check()
    print(drift_result)

    if do_train:
        train_result = run_train()
        print(train_result)

    print("=== Pipeline d'orchestration terminé ===")


if __name__ == "__main__":
    btc_pipeline(duration=30, do_train=False)