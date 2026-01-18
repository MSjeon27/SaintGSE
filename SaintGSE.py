#! /usr/bin/env python
# SaintGSE (ver. 1.0.1; https://github.com/MSjeon27/SaintGSE)
# Original SAINT project: https://github.com/somepago/saint
# Copyright 2020 - present, Facebook, Inc
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import glob
import os
import random
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from models.pretrainmodel import SaintGSE, Autoencoder
from models.augmentations import embed_data_nomask

# Silence warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS*")

# Parser
def build_parser(current_dir: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--AEmodel", default=os.path.join(current_dir, "datasets", "AE_100cycle_model.pth"))
    p.add_argument("--train", action="store_true")
    p.add_argument("--cv", action="store_true")
    p.add_argument("--dset", default=os.path.join(current_dir, "datasets", "AE_enrichment.tsv"))
    p.add_argument("--pathway", default=["all"], nargs="+", type=str)

    p.add_argument("--vision_dset", action="store_true")
    p.add_argument("--cont_embeddings", default="MLP", type=str, choices=["MLP", "Noemb", "pos_singleMLP"])
    p.add_argument("--embedding_size", default=32, type=int)
    p.add_argument("--transformer_depth", default=6, type=int)
    p.add_argument("--attention_heads", default=8, type=int)
    p.add_argument("--attention_dropout", default=0.1, type=float)
    p.add_argument("--ff_dropout", default=0.1, type=float)
    p.add_argument("--attentiontype", default="colrow", type=str,
                   choices=["col", "colrow", "row", "justmlp", "attn", "attnmlp"])

    p.add_argument("--optimizer", default="AdamW", type=str, choices=["AdamW", "Adam", "SGD"])
    p.add_argument("--scheduler", default="cosine", type=str, choices=["cosine", "linear"])

    p.add_argument("--lr", default=1e-5, type=float)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batchsize", default=256, type=int)
    p.add_argument("--run_name", default="testrun", type=str)
    p.add_argument("--set_seed", default=42, type=int)
    p.add_argument("--dset_seed", default=42, type=int)

    p.add_argument("--pretrain", action="store_true")
    p.add_argument("--pretrain_epochs", default=50, type=int)
    p.add_argument("--pt_tasks", default=["contrastive", "denoising"], type=str, nargs="*",
                   choices=["contrastive", "contrastive_sim", "denoising"])
    p.add_argument("--pt_aug", default=[], type=str, nargs="*", choices=["mixup", "cutmix"])
    p.add_argument("--pt_aug_lam", default=0.1, type=float)
    p.add_argument("--mixup_lam", default=0.3, type=float)

    p.add_argument("--train_mask_prob", default=0.0, type=float)
    p.add_argument("--mask_prob", default=0.0, type=float)

    p.add_argument("--ssl_avail_y", default=0, type=int)
    p.add_argument("--pt_projhead_style", default="diff", type=str, choices=["diff", "same", "nohead"])
    p.add_argument("--nce_temp", default=0.7, type=float)

    p.add_argument("--lam0", default=0.5, type=float)
    p.add_argument("--lam1", default=10.0, type=float)
    p.add_argument("--lam2", default=1.0, type=float)
    p.add_argument("--lam3", default=10.0, type=float)
    p.add_argument("--final_mlp_style", default="sep", type=str, choices=["common", "sep"])

    p.add_argument("--imbalance_mode", default="none", type=str,
                   choices=["none", "class_weight", "oversample", "focal"])
    p.add_argument("--focal_gamma", default=2.0, type=float)
    p.add_argument("--focal_alpha", default=0.25, type=float)

    p.add_argument("--ig_steps", default=256, type=int)
    p.add_argument("--ig_baseline", default="zero", type=str, choices=["mean", "zero"])
    p.add_argument("--predict", type=str, help="TSV file with gene-level expression for inference/IG")

    return p


# Utils
def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def load_pathway_list(current_dir: str) -> List[str]:
    fp = os.path.join(current_dir, "datasets", "pathway_list_in_DEG.txt")
    with open(fp, "r") as f:
        items = [line.rstrip() for line in f if line.strip()]
    return items


def select_target_pathways(current_dir: str, requested: List[str]) -> List[str]:
    all_paths = load_pathway_list(current_dir)
    if requested == ["all"]:
        return all_paths
    missing = [p for p in requested if p not in all_paths]
    if missing:
        raise ValueError(f"Unknown pathway(s): {missing}")
    return requested


def load_ae_state_dict(ae_ckpt: Union[dict, OrderedDict]) -> Dict[str, torch.Tensor]:
    if not isinstance(ae_ckpt, dict):
        return ae_ckpt

    if "model_state_dict" in ae_ckpt:
        return ae_ckpt["model_state_dict"]
    if "model" in ae_ckpt and isinstance(ae_ckpt["model"], (dict, OrderedDict)):
        return ae_ckpt["model"]
    if "state_dict" in ae_ckpt:
        return ae_ckpt["state_dict"]

    # fallback: keep tensor-like keys only
    return {k: v for k, v in ae_ckpt.items() if isinstance(v, torch.Tensor)}


def find_checkpoint_under_cwd(path_slug: str) -> str:
    cwd = os.getcwd()
    ckpt_names = ["best_infer.pt", "best.pt", "best_model.pt", "best.pth", "saint_gse_model.pth"]
    for nm in ckpt_names:
        pat = os.path.join(cwd, path_slug, "**", nm)
        hits = [h for h in glob.glob(pat, recursive=True) if os.path.isfile(h)]
        if hits:
            hits.sort(key=lambda p: p.count(os.sep))  # prefer shallow path
            return hits[0]
    raise FileNotFoundError(
        f"No checkpoint found under cwd for pathway slug '{path_slug}'. "
        f"Searched {ckpt_names} under ./{path_slug}/**"
    )

# IG calculator
class GeneToPathwayModel(torch.nn.Module):
    """
    Full chain: gene -> AE encoder -> latent standardization -> SAINT -> logit
    """
    def __init__(self, ae_model: Autoencoder, saint_wrapper: SaintGSE, device: torch.device):
        super().__init__()
        self.ae = ae_model
        self.saint = saint_wrapper.saint
        self.device = device

        # gene-level scaler stats (G,)
        self.register_buffer("gene_mean", ae_model.scaler_mean.clone())
        self.register_buffer("gene_std", ae_model.scaler_std.clone())

        # latent-level scaler stats (L,)
        cm, cs = saint_wrapper.opt.continuous_mean_std  # numpy [2, L]
        self.register_buffer("latent_mean", torch.tensor(cm, dtype=torch.float32))
        self.register_buffer("latent_std", torch.tensor(cs, dtype=torch.float32))

        self.vision_dset = saint_wrapper.opt.vision_dset

    def _standardize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return (x - mean) / std_safe

    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        """
        x_gene: (B, G)
        returns logits: (B,)
        """
        x_gene = x_gene.to(self.device).float()

        x_scaled = self._standardize(x_gene, self.gene_mean, self.gene_std) # (B, G)
        z = self.ae.encoder(x_scaled)   # (B, L)
        z_std = self._standardize(z, self.latent_mean, self.latent_std) # (B, L)

        B, _ = z_std.shape
        x_categ = torch.zeros(B, 1, dtype=torch.long, device=self.device)   # CLS
        x_cont = z_std  # (B, L)

        x_categ, x_categ_enc, x_cont_enc = embed_data_nomask(
            x_categ, x_cont, self.saint, vision_dset=self.vision_dset
        )

        reps = self.saint.transformer(x_categ_enc, x_cont_enc)  # (B, T, D)
        y = reps[:, 0, :]   # CLS (B, D)
        logits = self.saint.mlpfory(y).squeeze(-1)  # (B,)
        return logits


def integrated_gradients(
    model: GeneToPathwayModel,
    x_gene: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int,
) -> np.ndarray:
    if n_steps <= 0:
        raise ValueError("ig_steps must be >= 1 (ig_steps=0 leads to NaN IG).")

    device = model.device
    x_gene = x_gene.to(device).float().unsqueeze(0) # (1, G)
    baseline = baseline.to(device).float().unsqueeze(0) # (1, G)

    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device)
    total_grad = torch.zeros_like(x_gene)

    model.eval()
    for a in alphas:
        x = baseline + a * (x_gene - baseline)
        x.requires_grad_(True)
        logit = model(x).sum()
        grad = torch.autograd.grad(logit, x, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / float(n_steps)
    ig = (x_gene - baseline) * avg_grad
    return ig.squeeze(0).detach().cpu().numpy()

def train_mode(device: torch.device, opt: argparse.Namespace, current_dir: str) -> None:
    target_pathways = select_target_pathways(current_dir, opt.pathway)

    latent_df = pd.read_csv(opt.dset, sep="\t")
    for pathway in target_pathways:
        saint_gse = SaintGSE(device, latent_df, pathway, opt)
        saint_gse.run()

        os.makedirs(saint_gse.modelsave_path, exist_ok=True)
        save_path = os.path.join(saint_gse.modelsave_path, "saint_gse_model.pth")
        torch.save(
            {"model_state_dict": saint_gse.saint.state_dict(), "opt": vars(saint_gse.opt)},
            save_path,
        )
        print(f"[SAVE] {pathway} -> {save_path}")


def load_saint_for_inference(
    device: torch.device,
    opt: argparse.Namespace,
    latent_df: pd.DataFrame,
    pathway: str,
    ckpt_path: str,
) -> SaintGSE:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and ckpt.get("format") == "saintgse_infer_v1":
        opt_state = ckpt.get("opt_state", {})
        if isinstance(opt_state, argparse.Namespace):
            opt_state = vars(opt_state)
        if isinstance(opt_state, dict):
            for k, v in opt_state.items():
                setattr(opt, k, v)

        opt.continuous_mean_std = np.asarray(ckpt["continuous_mean_std"], dtype=np.float32)
        state = ckpt["model_state"]

        num_cont = int(ckpt["saint_init"]["num_continuous"])

        saint = SaintGSE(device, latent_df, pathway, opt, num_continuous_feature=num_cont)
        saint.saint.load_state_dict(state, strict=False)
        saint.saint.to(device)
        saint.eval()

        saint.best_threshold = float(ckpt.get("best_threshold", 0.5))
        a = ckpt.get("calib_a", None)
        b = ckpt.get("calib_b", None)
        saint.calib_a = torch.tensor(float(a), device=device) if a is not None else None
        saint.calib_b = torch.tensor(float(b), device=device) if b is not None else None
        return saint

    # legacy
    opt_loaded = ckpt.get("opt", {}) if isinstance(ckpt, dict) else {}
    if isinstance(opt_loaded, argparse.Namespace):
        opt_loaded = vars(opt_loaded)
    elif hasattr(opt_loaded, "__dict__") and not isinstance(opt_loaded, dict):
        opt_loaded = vars(opt_loaded)
    if isinstance(opt_loaded, dict):
        for k, v in opt_loaded.items():
            setattr(opt, k, v)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], (dict, OrderedDict)):
            state = ckpt["model"]
        else:
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        state = ckpt

    if isinstance(state, dict) and "norm.weight" in state:
        num_cont = int(state["norm.weight"].numel())
    else:
        num_cont = int(latent_df.shape[1])

    saint = SaintGSE(device, latent_df, pathway, opt, num_continuous_feature=num_cont)
    saint.saint.load_state_dict(state, strict=False)
    saint.saint.to(device)
    saint.eval()
    return saint


def predict_mode(device: torch.device, opt: argparse.Namespace, current_dir: str) -> None:
    if not opt.predict:
        raise ValueError("--predict is required in inference mode.")

    if opt.ig_steps <= 0:
        raise ValueError("ig_steps must be >= 1 (ig_steps=0 leads to NaN IG).")

    pathway = opt.pathway[0]
    pw_slug = safe_name(pathway)

    pred_df = pd.read_csv(opt.predict, sep="\t")
    if "Unnamed: 0" in pred_df.columns:
        pred_df = pred_df.set_index("Unnamed: 0")

    gene_names = pred_df.columns.tolist()
    input_dim = len(gene_names)
    latent_dim = 256

    # AE
    ae = Autoencoder(input_dim, latent_dim).to(device)
    ae_ckpt = torch.load(opt.AEmodel, map_location=device)
    ae.load_state_dict(load_ae_state_dict(ae_ckpt), strict=False)
    ae.eval()

    # latent features
    latent_df = pd.read_csv(opt.dset, sep="\t")
    if "Enrichment Results" in latent_df.columns:
        latent_df = latent_df.drop(columns=["Enrichment Results"])

    ckpt_path = find_checkpoint_under_cwd(pw_slug)
    opt.SGmodel = ckpt_path
    print(f"[INFO] Auto-selected SGmodel: {ckpt_path}")

    saint = load_saint_for_inference(device, opt, latent_df, pathway, ckpt_path)

    # full-chain model for IG
    gene_model = GeneToPathwayModel(ae, saint, device).to(device)
    gene_model.eval()

    # IG baseline
    if opt.ig_baseline == "zero":
        baseline_gene = torch.zeros(input_dim, dtype=torch.float32, device=device)
    else:
        baseline_gene = gene_model.gene_mean.detach().clone()

    pred_stem = os.path.splitext(os.path.basename(opt.predict))[0]
    out_dir = os.path.join(os.getcwd(), safe_name(pred_stem), pw_slug)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")

    sample_rows: List[Dict[str, object]] = []

    for idx, row in pred_df.iterrows():
        x_np = row.values.astype(np.float64, copy=False)
        if not np.isfinite(x_np).all():
            raise RuntimeError(f"Non-finite value in input row: {idx}")

        x_gene = torch.tensor(x_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            logit = gene_model(x_gene.unsqueeze(0)).squeeze(0)
            prob = torch.sigmoid(logit).item()
        print(f"\ncurrent index: {idx}")
        print(f"prediction_prob: {prob:.4f}")

        sample_rows.append({
            "sample_index": idx,
            "pathway": pathway,
            "logit": float(logit.item()),
            "prob": float(prob),
        })

        ig_attr = integrated_gradients(
            model=gene_model,
            x_gene=x_gene.detach().cpu(),
            baseline=baseline_gene.detach().cpu(),
            n_steps=int(opt.ig_steps),
        )

        ig_df = pd.DataFrame({
            "gene": gene_names,
            "log2FC": row.values.astype(float),
            "IG": ig_attr,
        })

        ig_df = ig_df[np.abs(ig_df["log2FC"].values) > 0.0].copy()
        ig_df["IG_abs"] = np.abs(ig_df["IG"].values)
        ig_df = ig_df.sort_values("IG_abs", ascending=False).drop(columns=["IG"])

        sample_tag = safe_name(idx)
        out_fp = os.path.join(out_dir, f"{sample_tag}_{pw_slug}_IG_gene_contributions.tsv")
        ig_df.to_csv(out_fp, sep="\t", index=False)
        print(f"IG gene-level contributions saved to '{out_fp}'")

    if sample_rows:
        sample_df = pd.DataFrame(sample_rows)
        score_fp = os.path.join(out_dir, f"{pw_slug}_IG_sample_scores.tsv")
        sample_df.to_csv(score_fp, sep="\t", index=False)
        print(f"Sample-level scores saved to '{score_fp}'")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = build_parser(current_dir)
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    fix_seed(int(opt.set_seed))

    if opt.train:
        train_mode(device, opt, current_dir)
    else:
        predict_mode(device, opt, current_dir)


if __name__ == "__main__":
    main()