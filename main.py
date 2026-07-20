"""
Orchestration de l'expérience « Ellipses » (Dorta et al., 2018, §5.1.2).

Enchaîne : config -> génération des datasets -> entraînement (NLL structurée) ->
évaluation (NLL, erreur de covariance, KL) -> figures.

Usage :
    ./venv/Scripts/python.exe main.py            # config par défaut (article)
    ./venv/Scripts/python.exe main.py --smoke    # petit run rapide (test de bout en bout)
    ./venv/Scripts/python.exe main.py --tag exp1 # nomme le run (dossier de sortie)

Chaque exécution écrit dans un dossier de run PROPRE et ISOLÉ :
    results/<tag>_<AAAAMMJJ-HHMMSS>/
qui contient tout ce que produit le run (checkpoint, métriques, figures). Deux
runs ne s'écrasent donc jamais, et les sorties restent triables par date.
"""

import argparse
import json
import os
from datetime import datetime

import torch

from dataset import EllipseDataset
from model import SparseCholeskyNet
from train import train
from eval import evaluate, make_figures, plot_history


def get_config(smoke=False):
    """Config par défaut = hyperparamètres de l'article ; `smoke` = version rapide."""
    cfg = {
        "image_size": 16,
        "f": 5,                 # voisinage 5×5
        "num_train": 15000,     # tailles article
        "num_test": 1000,
        "num_val": 1000,
        "epochs": 50,          # article
        "batch_size": 64,
        "lr": 1e-3,
        "base_ch": 32,
        "norm_input": True,
        "seed": 0,
        "results_root": "results",   # racine ; le dossier du run est créé dedans
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if smoke:
        cfg.update({
            "num_train": 512,
            "num_val": 128,
            "num_test": 128,
            "epochs": 3,
            "base_ch": 16,
        })
    return cfg


def make_run_dir(cfg, tag):
    """
    Crée un dossier de run PROPRE et unique : `results/<tag>_<horodatage>/`.

    L'horodatage garantit que deux runs successifs ne s'écrasent jamais. Le
    chemin est renvoyé et stocké dans `cfg["results_dir"]` (utilisé partout
    ensuite pour les checkpoints, figures et métriques).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "%s_%s" % (tag, timestamp)
    run_dir = os.path.join(cfg["results_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg["run_name"] = run_name
    cfg["results_dir"] = run_dir
    return run_dir


def build_datasets(cfg):
    """Crée les datasets train / val / test (val & test renvoient Sigma_true)."""
    common = dict(
        image_size=cfg["image_size"],
        # Les paramètres de covariance restent ceux par défaut du dataset
        # (noyau « lignes » l_along/l_across, overall_var, jitter).
    )
    print("Generation du dataset d'entrainement (%d exemples)..." % cfg["num_train"])
    train_ds = EllipseDataset(num_samples=cfg["num_train"], seed=cfg["seed"], **common)
    print("Generation du dataset de validation (%d exemples)..." % cfg["num_val"])
    val_ds = EllipseDataset(num_samples=cfg["num_val"], seed=cfg["seed"] + 1,
                            return_sigma=True, **common)
    print("Generation du dataset de test (%d exemples)..." % cfg["num_test"])
    test_ds = EllipseDataset(num_samples=cfg["num_test"], seed=cfg["seed"] + 2,
                             return_sigma=True, **common)
    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser(description="Experience ellipses (structured uncertainty).")
    parser.add_argument("--smoke", action="store_true",
                        help="petit run rapide de bout en bout (validation du pipeline).")
    parser.add_argument("--epochs", type=int, default=None, help="surcharge le nb d'epochs.")
    parser.add_argument("--tag", type=str, default=None,
                        help="prefixe du dossier de run (defaut : 'smoke' ou 'article').")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="chemin d'un checkpoint : evalue sans reentrainer.")
    args = parser.parse_args()

    cfg = get_config(smoke=args.smoke)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if cfg["device"] == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))
    else:
        print("Run sur CPU")

    # Dossier de run propre et horodaté (jamais d'ecrasement entre deux runs).
    tag = args.tag or ("smoke" if args.smoke else "article")
    run_dir = make_run_dir(cfg, tag)

    torch.manual_seed(cfg["seed"])
    print("Dossier du run :", run_dir)
    print("Config :", json.dumps(cfg, indent=2))

    train_ds, val_ds, test_ds = build_datasets(cfg)

    net = SparseCholeskyNet(
        image_size=cfg["image_size"], f=cfg["f"], base_ch=cfg["base_ch"]
    )

    if args.eval_only is not None:
        ckpt = torch.load(args.eval_only, map_location=cfg["device"])
        net.load_state_dict(ckpt["model_state"])
        print("Checkpoint charge :", args.eval_only)
    else:
        history = train(
            net, train_ds, val_ds,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            device=cfg["device"], image_size=cfg["image_size"], f=cfg["f"],
            norm_input=cfg["norm_input"], results_dir=cfg["results_dir"],
        )
        plot_history(history, results_dir=cfg["results_dir"])

        # Recharge le meilleur checkpoint avant l'évaluation finale.
        best_path = os.path.join(cfg["results_dir"], "best_model.pt")
        if os.path.exists(best_path):
            net.load_state_dict(torch.load(best_path, map_location=cfg["device"])["model_state"])

    # Évaluation finale sur le test.
    print("\nEvaluation sur le jeu de test...")
    metrics = evaluate(
        net, test_ds, device=cfg["device"], image_size=cfg["image_size"],
        f=cfg["f"], norm_input=cfg["norm_input"],
    )
    print("Metriques test :")
    print("  NLL moyenne              : %.4f" % metrics["nll"])
    print("  Erreur Frobenius relative: %.4f" % metrics["frobenius_rel"])
    print("  KL(vraie || predite)     : %.4f" % metrics["kl"])

    with open(os.path.join(cfg["results_dir"], "metrics.json"), "w") as fp:
        json.dump({"config": cfg, "metrics": metrics}, fp, indent=2)

    make_figures(
        net, test_ds, device=cfg["device"], image_size=cfg["image_size"],
        f=cfg["f"], norm_input=cfg["norm_input"], results_dir=cfg["results_dir"],
    )
    print("\nTermine. Sorties dans :", cfg["results_dir"])


if __name__ == "__main__":
    main()
