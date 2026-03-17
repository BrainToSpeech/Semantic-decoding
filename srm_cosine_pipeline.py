import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainiak.funcalign.srm import SRM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_std(x, eps=1e-6):
    return np.maximum(x, eps).astype(np.float32)


def run_epoch(model, loader, optimizer, loss_type, device, train):
    model.train(train)
    loss_sum = 0.0
    cos_sum = 0.0
    mse_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(xb)
        if loss_type == "mse":
            loss = F.mse_loss(pred, yb)
        elif loss_type == "cosine_alignment":
            loss = (1 - F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(yb, dim=-1), dim=-1)).mean()
        else:
            raise ValueError(f"Unsupported loss.type: {loss_type}")
        if train:
            loss.backward()
            optimizer.step()
        b = xb.size(0)
        loss_sum += float(loss.item()) * b
        cos_sum += float(F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(yb, dim=-1), dim=-1).sum().item())
        mse_sum += float(F.mse_loss(pred, yb, reduction="none").reshape(b, -1).mean(dim=1).sum().item())
        n += b
    n = max(n, 1)
    return {"loss": loss_sum / n, "cos": cos_sum / n, "mse": mse_sum / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["experiment"].get("cuda_visible_devices", "0"))
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = os.path.abspath(cfg["experiment"]["output_root"])
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    with open(cfg["data"]["file_path"], "rb") as f:
        raw = pickle.load(f)
    trial = raw["word_data"]
    hg = np.asarray(trial["highgamma"], dtype=np.float32)
    onsets = np.asarray(trial["onset"], dtype=np.float64)
    words = np.load(cfg["data"]["embedding_path"]).astype(np.float32)
    sig = np.asarray(raw["significant_elecs"]).astype(bool)
    subj = np.asarray(
        [
            int(str(x)[7:]) if str(x).lower().startswith("subject") else int(x)
            for x in trial["subj"]
        ],
        dtype=np.int64,
    )

    fs = int(cfg["data"]["fs"])
    win = max(int(round(float(cfg["data"]["window_ms"]) * fs / 1000.0)), 1)
    n = min(len(onsets), len(words))
    start = np.round(onsets[:n] * fs).astype(np.int64)
    valid = (start >= 0) & (start + win <= hg.shape[0])
    start = start[valid]
    text_raw = words[:n][valid]

    n_samples = len(text_raw)
    train_end = min(int(cfg["data"]["train_end"]), n_samples)
    val_end = min(int(cfg["data"]["val_end"]), n_samples)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)

    if cfg["text_embedding"].get("use_pca", True):
        centered = text_raw - np.mean(text_raw[train_idx], axis=0, keepdims=True)
        pca_dim = min(int(cfg["text_embedding"].get("pca_dim", 50)), centered.shape[1], max(len(train_idx), 1))
        pca = PCA(n_components=pca_dim)
        pca.fit(centered[train_idx])
        text_emb = pca.transform(centered).astype(np.float32)
        if cfg["text_embedding"].get("l2_normalize", True):
            text_emb = normalize(text_emb, axis=1, norm="l2").astype(np.float32)
    else:
        pca = None
        text_emb = text_raw.astype(np.float32)

    print("device:", device)
    print("n_valid_samples:", n_samples)
    print("text_raw shape:", text_raw.shape)
    print("text_emb shape:", text_emb.shape)
    if pca is not None:
        print("text PCA explained variance ratio sum:", float(np.sum(pca.explained_variance_ratio_)))

    available_subjects = sorted(np.unique(subj[sig]).tolist())
    subject_list = [int(s) for s in cfg["data"]["subject_list"]]
    missing = [s for s in subject_list if s not in available_subjects]
    if missing:
        raise ValueError(f"Unavailable subjects in subject_list: {missing}")

    all_needed = set(subject_list)
    srm_cfg = cfg.get("srm", {})
    fit_subjects = []
    if srm_cfg.get("enabled", False):
        fit_subjects = [int(s) for s in srm_cfg.get("fit_subject_list", subject_list)]
        fit_subjects = [s for s in fit_subjects if s in available_subjects]
        all_needed |= set(fit_subjects)

    csum = np.concatenate(
        [np.zeros((1, hg.shape[1]), dtype=np.float32), np.cumsum(hg, axis=0, dtype=np.float32)],
        axis=0,
    )
    targets = {}
    for sid in sorted(all_needed):
        mask = (subj == sid) & sig
        if not np.any(mask):
            raise ValueError(f"No significant electrodes found for subject {sid}")
        x = (csum[start + win][:, mask] - csum[start][:, mask]) / float(win)
        targets[sid] = x.astype(np.float32)

    srm_params = None
    if srm_cfg.get("enabled", False) and len(fit_subjects) >= 2:
        train_data = []
        srm_stats = {}
        min_ch = None
        for sid in fit_subjects:
            x = targets[sid]
            mu = np.mean(x[train_idx], axis=0).astype(np.float32)
            std = safe_std(np.std(x[train_idx], axis=0).astype(np.float32))
            xz = np.nan_to_num((x[train_idx] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            train_data.append(xz.T)
            srm_stats[sid] = (mu, std)
            min_ch = x.shape[1] if min_ch is None else min(min_ch, x.shape[1])
        features = min(int(srm_cfg["features"]), int(min_ch), max(len(train_idx) - 1, 1))
        srm = SRM(n_iter=int(srm_cfg.get("n_iter", 100)), features=features)
        srm.fit(train_data)
        srm_params = {}
        for i, sid in enumerate(fit_subjects):
            mu, std = srm_stats[sid]
            srm_params[sid] = {"w": np.asarray(srm.w_[i], dtype=np.float32), "mu": mu, "std": std, "mode": "fit_subject"}
        if srm_cfg.get("generalize_unseen_subjects", True):
            unseen = [s for s in subject_list if s not in srm_params]
            if unseen:
                print("Unseen subjects for SRM generalization:", unseen)
            for sid in unseen:
                x = targets[sid]
                mu = np.mean(x[train_idx], axis=0).astype(np.float32)
                std = safe_std(np.std(x[train_idx], axis=0).astype(np.float32))
                xz = np.nan_to_num((x[train_idx] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                try:
                    w = srm.transform_subject(xz.T)
                    mode = "generalized_transform_subject"
                except Exception:
                    u, _, vh = np.linalg.svd(xz.T @ srm.s_.T, full_matrices=False)
                    w = u @ vh
                    mode = "generalized_procrustes"
                srm_params[sid] = {"w": np.asarray(w, dtype=np.float32), "mu": mu, "std": std, "mode": mode}

    for sid in subject_list:
        out_dir = os.path.join(output_root, f"subject_{sid}")
        model_dir = os.path.join(out_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        y = targets[sid].copy()
        srm_mode = "disabled"
        srm_applied = False
        if srm_params is not None and sid in srm_params:
            p = srm_params[sid]
            yz = (y - p["mu"][None, :]) / p["std"][None, :]
            y = ((yz @ p["w"]) @ p["w"].T) * p["std"][None, :] + p["mu"][None, :]
            y = y.astype(np.float32)
            srm_mode = p["mode"]
            srm_applied = True

        if cfg["data"].get("zscore_target", True):
            mu = np.mean(y[train_idx], axis=0, keepdims=True).astype(np.float32)
            std = safe_std(np.std(y[train_idx], axis=0, keepdims=True).astype(np.float32))
            y = ((y - mu) / std).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(text_emb[train_idx]), torch.from_numpy(y[train_idx]))
        val_ds = TensorDataset(torch.from_numpy(text_emb[val_idx]), torch.from_numpy(y[val_idx]))
        test_ds = TensorDataset(torch.from_numpy(text_emb[test_idx]), torch.from_numpy(y[test_idx]))
        train_loader = DataLoader(train_ds, batch_size=int(cfg["data"].get("batch_size", 64)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(cfg["data"].get("batch_size", 64)), shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=int(cfg["data"].get("batch_size", 64)), shuffle=False)

        mcfg = cfg["model"]
        if str(mcfg.get("arch", "mlp")).lower() == "linear":
            model = nn.Linear(text_emb.shape[1], y.shape[1])
        else:
            hidden = int(mcfg.get("hidden_dim", 128))
            layers = []
            in_dim = text_emb.shape[1]
            for _ in range(max(int(mcfg.get("num_layers", 2)) - 1, 0)):
                layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(float(mcfg.get("dropout_rate", 0.5)))]
                in_dim = hidden
            layers.append(nn.Linear(in_dim, y.shape[1]))
            model = nn.Sequential(*layers)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"].get("lr", 5e-4)),
            weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        )
        loss_type = str(cfg["loss"]["type"]).lower()

        best_val_loss = float("inf")
        best_path = os.path.join(model_dir, "best_val_model.pth")
        for epoch in range(1, int(cfg["train"]["num_epochs"]) + 1):
            tr = run_epoch(model, train_loader, optimizer, loss_type, device, True)
            va = run_epoch(model, val_loader, optimizer, loss_type, device, False)
            print(
                f"[subject_{sid}][epoch {epoch:03d}] "
                f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                f"val_cos={va['cos']:.4f} val_mse={va['mse']:.4f}"
            )
            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        te = run_epoch(model, test_loader, optimizer, loss_type, device, False)
        print(f"[TEST][subject_{sid}] cos={te['cos']:.6f} mse={te['mse']:.6f} loss={te['loss']:.6f}")

        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subject": sid,
                    "srm_applied_to_subject": int(srm_applied),
                    "srm_mode": srm_mode,
                    "loss_type": loss_type,
                    "cosine_similarity": float(te["cos"]),
                    "mse": float(te["mse"]),
                    "loss": float(te["loss"]),
                    "dataset_size": int(len(text_emb)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
