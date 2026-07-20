"""
Évaluation : métriques (NLL, erreur de covariance, KL) et figures.

À l'inverse de l'entraînement, l'évaluation A ACCÈS à la vraie covariance
`Sigma_true` (reconstruite à la volée par le dataset) et la compare à la
covariance prédite `Sigma_pred = (L L^T)^{-1}`.

Métriques :
- NLL        : vraisemblance négative gaussienne structurée moyenne (comme la loss).
- Frobenius  : erreur relative ||Sigma_pred - Sigma_true||_F / ||Sigma_true||_F.
- KL         : KL( N(0, Sigma_true) || N(0, Sigma_pred) ), moyenne, qui mesure
               à quel point la loi prédite explique le vrai bruit corrélé.

Figures :
- comparaison de la « carte de corrélation » d'un pixel (une colonne de Sigma
  remise en image) : vraie vs prédite -> montre si les lignes de corrélation
  prédites s'alignent bien avec l'ellipse ;
- échantillons de bruit tirés de la loi prédite vs de la vraie loi.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from loss import (
    build_neighbor_indices,
    build_L_dense,
    structured_gaussian_nll,
)
from train import normalize_mu


@torch.no_grad()
def evaluate(net, dataset, device="cpu", image_size=16, f=5, norm_input=True,
             batch_size=64):
    """
    Calcule les métriques moyennes sur `dataset` (qui doit renvoyer Sigma_true).

    Retour :
        metrics : dict {'nll', 'frobenius_rel', 'kl'}
    """
    device = torch.device(device)
    net = net.to(device).eval()
    neighbor_idx, mask = build_neighbor_indices(image_size, f, device=device)
    n = image_size * image_size

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sum_nll = sum_frob = sum_kl = 0.0
    count = 0

    for batch in loader:
        mu = batch["mu"].to(device)
        x = batch["x"].to(device)
        Sigma_true = batch["Sigma_true"].to(device).double()
        B = mu.shape[0]

        log_diag, offdiag = net(normalize_mu(mu, norm_input))

        # NLL structurée sur le résidu observé.
        nll = structured_gaussian_nll(log_diag, offdiag, x - mu, neighbor_idx, mask,
                                      mean_batch=False)
        sum_nll += nll.sum().item()

        # Covariance / précision prédites (double précision pour l'inversion).
        L = build_L_dense(log_diag.double(), offdiag.double(), neighbor_idx, mask)
        Lambda = L @ L.transpose(1, 2)
        Sigma_pred = torch.linalg.inv(Lambda)

        # Erreur de Frobenius relative.
        num = torch.linalg.matrix_norm(Sigma_pred - Sigma_true, ord="fro", dim=(1, 2))
        den = torch.linalg.matrix_norm(Sigma_true, ord="fro", dim=(1, 2))
        sum_frob += (num / den).sum().item()

        # KL( N(0,Sigma_true) || N(0,Sigma_pred) )
        #   = 0.5 * [ tr(Lambda_pred Sigma_true) - n + log|Sigma_pred| - log|Sigma_true| ]
        tr_term = torch.einsum("bij,bji->b", Lambda, Sigma_true)
        logdet_pred = -2.0 * log_diag.double().sum(dim=1)       # log|Sigma_pred|
        logdet_true = torch.logdet(Sigma_true)                  # log|Sigma_true|
        kl = 0.5 * (tr_term - n + logdet_pred - logdet_true)
        sum_kl += kl.sum().item()

        count += B

    return {
        "nll": sum_nll / count,
        "frobenius_rel": sum_frob / count,
        "kl": sum_kl / count,
    }


@torch.no_grad()
def sample_from_precision(L, num_samples=1):
    """
    Tire des échantillons `eps ~ N(0, Sigma)` avec `Sigma = (L L^T)^{-1}`.

    Si `Lambda = L L^T`, alors `eps = L^{-T} z` (z ~ N(0, I)) a bien pour
    covariance `L^{-T} L^{-1} = Sigma`. On résout donc `L^T eps = z`
    (système triangulaire supérieur), sans inverser explicitement.

    Args :
        L : [B, n, n] Cholesky (triangulaire inf.) de la précision.
    Retour :
        eps : [B, num_samples, n]
    """
    B, n, _ = L.shape
    LT = L.transpose(1, 2)  # triangulaire supérieure
    z = torch.randn(B, n, num_samples, dtype=L.dtype, device=L.device)
    eps = torch.linalg.solve_triangular(LT, z, upper=True)  # [B, n, num_samples]
    return eps.transpose(1, 2)  # [B, num_samples, n]


@torch.no_grad()
def make_figures(net, dataset, device="cpu", image_size=16, f=5, norm_input=True,
                 num_examples=4, results_dir="results", seed=0):
    """
    Sauvegarde deux figures comparant loi prédite et vraie loi sur quelques
    exemples : cartes de corrélation d'un pixel, et échantillons de bruit.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Figures non generees (matplotlib indisponible ?) :", e)
        return

    os.makedirs(results_dir, exist_ok=True)
    device = torch.device(device)
    net = net.to(device).eval()
    neighbor_idx, mask = build_neighbor_indices(image_size, f, device=device)
    S = image_size
    n = S * S
    torch.manual_seed(seed)

    # Sélection d'exemples.
    idxs = list(range(num_examples))
    mus, Sig_true = [], []
    for i in idxs:
        item = dataset[i]
        mus.append(item["mu"])
        Sig_true.append(item["Sigma_true"])
    mu = torch.stack(mus).to(device)
    Sigma_true = torch.stack(Sig_true).to(device).double()

    log_diag, offdiag = net(normalize_mu(mu, norm_input))
    L = build_L_dense(log_diag.double(), offdiag.double(), neighbor_idx, mask)
    Lambda = L @ L.transpose(1, 2)
    Sigma_pred = torch.linalg.inv(Lambda)

    # --- Figure 1 : carte de corrélation d'un pixel central ---
    # On choisit, pour chaque exemple, le pixel de plus forte intensité de mu
    # (au coeur de l'ellipse), et on affiche la colonne correspondante de Sigma.
    fig, axes = plt.subplots(3, num_examples, figsize=(3 * num_examples, 9))
    for k in range(num_examples):
        mu_img = mu[k].cpu().numpy().reshape(S, S)
        p = int(np.argmax(mu_img))  # pixel de reference (coeur de l'ellipse)

        col_true = Sigma_true[k, :, p].cpu().numpy().reshape(S, S)
        col_pred = Sigma_pred[k, :, p].cpu().numpy().reshape(S, S)
        vmax = max(np.abs(col_true).max(), np.abs(col_pred).max()) + 1e-12

        axes[0, k].imshow(mu_img, cmap="gray")
        axes[0, k].set_title(f"mu #{k}")
        axes[1, k].imshow(col_true, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[1, k].set_title("Sigma_true[:, p]")
        axes[2, k].imshow(col_pred, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[2, k].set_title("Sigma_pred[:, p]")
        for row in range(3):
            axes[row, k].axis("off")
    plt.tight_layout()
    out1 = os.path.join(results_dir, "eval_correlation_maps.png")
    plt.savefig(out1, dpi=150)
    plt.close()

    # --- Figure 2 : échantillons de bruit (vrai vs prédit) ---
    eps_pred = sample_from_precision(L, num_samples=1)[:, 0, :]  # [B, n]
    # Vrai bruit : Cholesky de Sigma_true puis eps = C z.
    Ctrue = torch.linalg.cholesky(Sigma_true)
    z = torch.randn(num_examples, n, 1, dtype=torch.float64, device=device)
    eps_true = (Ctrue @ z)[:, :, 0]

    fig, axes = plt.subplots(3, num_examples, figsize=(3 * num_examples, 9))
    for k in range(num_examples):
        mu_img = mu[k].cpu().numpy().reshape(S, S)
        et = eps_true[k].cpu().numpy().reshape(S, S)
        ep = eps_pred[k].cpu().numpy().reshape(S, S)
        vmax = max(np.abs(et).max(), np.abs(ep).max()) + 1e-12

        axes[0, k].imshow(mu_img, cmap="gray")
        axes[0, k].set_title(f"mu #{k}")
        axes[1, k].imshow(et, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[1, k].set_title("eps ~ vraie loi")
        axes[2, k].imshow(ep, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[2, k].set_title("eps ~ loi predite")
        for row in range(3):
            axes[row, k].axis("off")
    plt.tight_layout()
    out2 = os.path.join(results_dir, "eval_samples.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    print("figures sauvegardees :", out1, "|", out2)


def plot_history(history, results_dir="results"):
    """Trace les courbes de NLL train/val."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Courbe non generee (matplotlib indisponible ?) :", e)
        return
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_nll"], label="train")
    plt.plot(history["val_nll"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.title("Courbe d'apprentissage (NLL)")
    plt.tight_layout()
    out = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("courbe sauvegardee :", out)
