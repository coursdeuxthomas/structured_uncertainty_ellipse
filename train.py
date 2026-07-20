"""
Boucle d'entraînement + validation + checkpoints (ellipses).

On optimise UNIQUEMENT la NLL gaussienne structurée (`loss.structured_gaussian_nll`)
sur `(mu, x)` : la vraie covariance n'est jamais utilisée à l'entraînement.

Le réseau reçoit `mu` (image de l'ellipse) et prédit la Cholesky creuse de la
précision du résidu `epsilon = x - mu` ; la loss mesure la vraisemblance du
résidu observé sous cette précision.
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from loss import build_neighbor_indices, structured_gaussian_nll


def normalize_mu(mu, do_norm):
    """Normalise l'entrée du réseau dans ~[-1, 1] (mu est dans [0, 1])."""
    return mu * 2.0 - 1.0 if do_norm else mu


def run_epoch(net, loader, neighbor_idx, mask, optimizer, device, norm_input, train):
    """Fait une passe (train ou éval) ; renvoie la NLL moyenne sur les exemples."""
    net.train(train)
    total_nll = 0.0
    total_count = 0

    torch.set_grad_enabled(train)
    for batch in loader:
        mu = batch["mu"].to(device)
        x = batch["x"].to(device)
        residual = x - mu
        B = mu.shape[0]

        log_diag, offdiag = net(normalize_mu(mu, norm_input))
        loss = structured_gaussian_nll(log_diag, offdiag, residual, neighbor_idx, mask)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

        total_nll += loss.item() * B
        total_count += B

    torch.set_grad_enabled(True)
    return total_nll / total_count


def train(
    net,
    train_dataset,
    val_dataset,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    device="cpu",
    image_size=16,
    f=5,
    norm_input=True,
    results_dir="results",
    num_workers=0,
    log_every=1,
):
    """
    Entraîne le réseau et sauvegarde le meilleur checkpoint (selon la NLL val).

    Retour :
        history : dict avec les listes 'train_nll' et 'val_nll' par epoch.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device(device)
    net = net.to(device)

    neighbor_idx, mask = build_neighbor_indices(image_size, f, device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = {"train_nll": [], "val_nll": []}
    best_val = float("inf")
    best_path = os.path.join(results_dir, "best_model.pt")

    print("Debut de l'entrainement : %d epochs, %d exemples, batch %d, lr %.1e"
          % (epochs, len(train_dataset), batch_size, lr))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_nll = run_epoch(net, train_loader, neighbor_idx, mask, optimizer,
                              device, norm_input, train=True)
        val_nll = run_epoch(net, val_loader, neighbor_idx, mask, optimizer,
                            device, norm_input, train=False)

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        if val_nll < best_val:
            best_val = val_nll
            torch.save(
                {
                    "model_state": net.state_dict(),
                    "epoch": epoch,
                    "val_nll": val_nll,
                    "config": {"image_size": image_size, "f": f, "norm_input": norm_input},
                },
                best_path,
            )

        if epoch % log_every == 0 or epoch == epochs:
            dt = time.time() - t0
            flag = "  <- meilleur" if val_nll == best_val else ""
            print("epoch %3d/%d | train NLL %10.3f | val NLL %10.3f | %.1fs%s"
                  % (epoch, epochs, train_nll, val_nll, dt, flag))

    print("Entrainement termine. Meilleure NLL val : %.3f (%s)" % (best_val, best_path))
    return history
