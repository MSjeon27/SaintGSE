from __future__ import annotations
from .model import *


import os
import json
import copy
import random
import ast
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()

        # scaler stats (gene-level)
        self.register_buffer("scaler_mean", torch.zeros(input_dim))
        self.register_buffer("scaler_std", torch.ones(input_dim))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.training and self.noise_std and self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def preprocess_pred(self, df: pd.DataFrame):
        """
        df: (B, G) raw gene-level inputs
        return: (B, G) standardized using training scaler stats
        """
        X = df.values.astype(np.float32)    # (B, G)
        X_t = torch.from_numpy(X).to(self.scaler_mean.device)   # (B, G)

        mean = self.scaler_mean
        std = self.scaler_std
        std = torch.where(std == 0, torch.ones_like(std), std)

        X_scaled = (X_t - mean) / std
        return X_scaled.cpu().numpy()


# SAINT
class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = []
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=0,
        attn_dropout=0.,
        ff_dropout=0.,
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col',
        final_mlp_style='common',
        y_dim=2
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * self.num_continuous)
            nfeats = self.num_categories + self.num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * self.num_continuous)
            nfeats = self.num_categories + self.num_continuous
        else:
            raise ValueError("cont_embeddings must be 'MLP' or 'pos_singleMLP' for classification performance")

        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )
        else:
            raise ValueError(f"Unknown attentiontype: {attentiontype}")

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])
        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP(
            [dim * (self.num_continuous + self.num_categories),
             6 * dim * (self.num_continuous + self.num_categories) // 5,
             dim * (self.num_continuous + self.num_categories) // 2]
        )
        self.pt_mlp2 = simple_MLP(
            [dim * (self.num_continuous + self.num_categories),
             6 * dim * (self.num_continuous + self.num_categories) // 5,
             dim * (self.num_continuous + self.num_categories) // 2]
        )

    def forward(self, x_categ, x_cont):
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:, :self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories:, :])
        return cat_outs, con_outs


# SaintGSE
class SaintGSE(nn.Module):
    def __init__(
        self,
        device,
        latent_df: Optional[pd.DataFrame],
        target_pathway: str,
        opt,
        num_continuous_feature: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.latent_df = latent_df
        self.pathway = target_pathway
        self.opt = opt

        # defaults
        if not hasattr(self.opt, "optimizer"):
            self.opt.optimizer = "AdamW"
        if not hasattr(self.opt, "lr"):
            self.opt.lr = 1e-4
        if not hasattr(self.opt, "epochs"):
            self.opt.epochs = 100
        if not hasattr(self.opt, "batchsize"):
            self.opt.batchsize = 256
        if not hasattr(self.opt, "early_patience"):
            self.opt.early_patience = 50
        if not hasattr(self.opt, "valid_ratio"):
            self.opt.valid_ratio = 0.15
        if not hasattr(self.opt, "dset_seed"):
            self.opt.dset_seed = 1
        if not hasattr(self.opt, "set_seed"):
            self.opt.set_seed = 1
        if not hasattr(self.opt, "vision_dset"):
            self.opt.vision_dset = False
        if not hasattr(self.opt, "out_dir"):
            self.opt.out_dir = getattr(self.opt, "current_dir", ".")
        if not hasattr(self.opt, "embedding_size"):
            self.opt.embedding_size = 32
        if not hasattr(self.opt, "transformer_depth"):
            self.opt.transformer_depth = 2
        if not hasattr(self.opt, "attention_heads"):
            self.opt.attention_heads = 8
        if not hasattr(self.opt, "attention_dropout"):
            self.opt.attention_dropout = 0.1
        if not hasattr(self.opt, "ff_dropout"):
            self.opt.ff_dropout = 0.1
        if not hasattr(self.opt, "cont_embeddings"):
            self.opt.cont_embeddings = "MLP"
        if not hasattr(self.opt, "attentiontype"):
            self.opt.attentiontype = "col"
        if not hasattr(self.opt, "final_mlp_style"):
            self.opt.final_mlp_style = "common"

        re_path = self.pathway.replace(' ', '_') if self.pathway else "NA"
        self.modelsave_path = os.path.join(
            getattr(self.opt, "current_dir", "."),
            'datasets',
            f'bestmodels/binary/{re_path}/testrun'
        )

        if num_continuous_feature is None:
            if latent_df is not None:
                num_continuous_feature = int(latent_df.shape[-1] - 1)  # except label column
            else:
                num_continuous_feature = 256

        categories_init = np.array([1]).astype(int)

        # dynamic hyperparam tweaks
        if num_continuous_feature > 100:
            self.opt.embedding_size = min(8, int(self.opt.embedding_size))
            self.opt.batchsize = min(64, int(self.opt.batchsize))

        if self.opt.attentiontype != 'col':
            self.opt.transformer_depth = 1
            self.opt.attention_heads = min(4, int(self.opt.attention_heads))
            self.opt.attention_dropout = 0.8
            self.opt.embedding_size = min(32, int(self.opt.embedding_size))
            self.opt.ff_dropout = min(float(self.opt.ff_dropout), 0.4)

        self.saint = SAINT(
            categories=categories_init,
            num_continuous=int(num_continuous_feature),
            dim=int(self.opt.embedding_size),
            dim_out=1,
            depth=int(self.opt.transformer_depth),
            heads=int(self.opt.attention_heads),
            attn_dropout=float(self.opt.attention_dropout),
            ff_dropout=float(self.opt.ff_dropout),
            mlp_hidden_mults=(4, 2),
            cont_embeddings=self.opt.cont_embeddings,
            attentiontype=self.opt.attentiontype,
            final_mlp_style=self.opt.final_mlp_style,
            y_dim=1,
        )

        # post-hoc calibration params
        self.calib_a = None
        self.calib_b = None

        # training-time feature indices
        self._train_cat_idxs = None
        self._train_con_idxs = None

    # ckpt payload
    def _pack_infer_ckpt(
        self,
        *,
        model_state,
        continuous_mean_std,
        best_threshold,
        cat_idxs,
        con_idxs,
        calib_a,
        calib_b
    ):
        opt_keys = [
            "vision_dset", "embedding_size", "transformer_depth", "attention_heads",
            "attention_dropout", "ff_dropout", "cont_embeddings", "attentiontype",
            "final_mlp_style"
        ]
        opt_state = {k: getattr(self.opt, k) for k in opt_keys if hasattr(self.opt, k)}

        saint_init = {
            "categories": [1],
            "num_continuous": int(self.saint.num_continuous),
            "dim": int(self.saint.dim),
            "depth": int(getattr(self.opt, "transformer_depth", 1)),
            "heads": int(getattr(self.opt, "attention_heads", 1)),
            "attn_dropout": float(getattr(self.opt, "attention_dropout", 0.0)),
            "ff_dropout": float(getattr(self.opt, "ff_dropout", 0.0)),
            "cont_embeddings": getattr(self.opt, "cont_embeddings", "MLP"),
            "attentiontype": getattr(self.opt, "attentiontype", "col"),
            "final_mlp_style": getattr(self.opt, "final_mlp_style", "common"),
            "y_dim": 1,
            "dim_out": 1,
        }

        return {
            "format": "saintgse_infer_v1",
            "pathway": self.pathway,
            "model_state": model_state,
            "saint_init": saint_init,
            "opt_state": opt_state,
            "continuous_mean_std": np.asarray(continuous_mean_std, dtype=np.float32),
            "best_threshold": float(best_threshold),
            "cat_idxs": list(cat_idxs),
            "con_idxs": list(con_idxs),
            "calib_a": float(calib_a) if calib_a is not None else None,
            "calib_b": float(calib_b) if calib_b is not None else None,
        }

    def load_infer_ckpt(self, ckpt_path: str):
        payload = torch.load(ckpt_path, map_location=self.device)
        if not isinstance(payload, dict) or payload.get("format", "") != "saintgse_infer_v1":
            raise ValueError(f"Invalid inference ckpt format: {ckpt_path}")

        si = payload["saint_init"]
        self.opt.vision_dset = payload.get("opt_state", {}).get("vision_dset", getattr(self.opt, "vision_dset", False))
        for k, v in payload.get("opt_state", {}).items():
            setattr(self.opt, k, v)

        self.saint = SAINT(
            categories=np.array(si["categories"]).astype(int),
            num_continuous=int(si["num_continuous"]),
            dim=int(si["dim"]),
            depth=int(si["depth"]),
            heads=int(si["heads"]),
            attn_dropout=float(si["attn_dropout"]),
            ff_dropout=float(si["ff_dropout"]),
            cont_embeddings=si["cont_embeddings"],
            attentiontype=si["attentiontype"],
            final_mlp_style=si["final_mlp_style"],
            y_dim=int(si["y_dim"]),
            dim_out=int(si["dim_out"]),
        ).to(self.device)

        self.saint.load_state_dict(payload["model_state"])
        self.saint.eval()

        self.opt.continuous_mean_std = payload["continuous_mean_std"]
        self.best_threshold = float(payload.get("best_threshold", 0.5))

        self._train_cat_idxs = list(payload.get("cat_idxs", []))
        self._train_con_idxs = list(payload.get("con_idxs", []))

        a = payload.get("calib_a", None)
        b = payload.get("calib_b", None)
        if (a is None) or (b is None):
            self.calib_a, self.calib_b = None, None
        else:
            self.calib_a = torch.tensor(float(a), dtype=torch.float32, device=self.device)
            self.calib_b = torch.tensor(float(b), dtype=torch.float32, device=self.device)

        return self

    # utilities
    def _apply_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        if (self.calib_a is None) or (self.calib_b is None):
            return torch.sigmoid(logits)
        return torch.sigmoid(self.calib_a * logits + self.calib_b)

    def run(self):
        """
        Training entry.
        - CV mode: opt.cv=True
        - Single split: opt.cv=False
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
        from sklearn.metrics import (
            average_precision_score, precision_recall_curve, f1_score,
            roc_auc_score, accuracy_score
        )

        from .data_preprocess import data_prep, DataSetCatCon
        from .augmentations import embed_data_mask

        # Helpers
        def set_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        def _collate_to_tensors(batch):
            x_cat, x_con, y, m_cat, m_con = zip(*batch)
            x_cat = torch.from_numpy(np.stack(x_cat, axis=0)).long()
            x_con = torch.from_numpy(np.stack(x_con, axis=0)).float()
            y     = torch.from_numpy(np.stack(y, axis=0)).float()
            m_cat = torch.from_numpy(np.stack(m_cat, axis=0)).long()
            m_con = torch.from_numpy(np.stack(m_con, axis=0)).long()
            return x_cat, x_con, y, m_cat, m_con

        def _parse_enrichment_cell(cell):
            if isinstance(cell, (list, tuple, set)):
                return list(cell)
            if not isinstance(cell, str):
                return []
            s = cell.strip()
            if not s:
                return []
            if (s[0] in "[(" and s[-1] in "])") or (s[0] == "{" and s[-1] == "}"):
                try:
                    obj = ast.literal_eval(s)
                    if isinstance(obj, (list, tuple, set)):
                        return list(obj)
                except Exception:
                    pass
            for sep in ["|", ";", ","]:
                if sep in s:
                    return [t.strip() for t in s.split(sep) if t.strip()]
            return [s]

        def _label_from_enrichment(cell, pathway: str) -> int:
            return 1 if pathway in _parse_enrichment_cell(cell) else 0

        def _y_from_split(yobj):
            arr = yobj["data"] if isinstance(yobj, dict) else yobj
            return np.asarray(arr).reshape(-1).astype(np.int64)

        def _safe_auroc(y, p):
            return float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")

        def _sigmoid_np(x):
            x = np.asarray(x, dtype=np.float64)
            return 1.0 / (1.0 + np.exp(-x))

        # Labeling
        if self.latent_df is None:
            raise ValueError("latent_df is required for training.")
        if "Enrichment Results" not in self.latent_df.columns:
            raise KeyError("Missing column: 'Enrichment Results'")

        df = self.latent_df.copy()
        df[self.pathway] = df["Enrichment Results"].apply(lambda x: _label_from_enrichment(x, self.pathway))
        df = df.drop(columns=["Enrichment Results"])

        print(f"SaintGSE for pathway: {self.pathway}")
        print(f"df shape: {df.shape}")

        vision_dset = self.opt.vision_dset
        self.saint.vision_dset = vision_dset

        base_dir = os.path.join(self.opt.out_dir, str(self.pathway).replace(" ", "_"))
        os.makedirs(base_dir, exist_ok=True)

        # Initial split (get cat_idxs/con_idxs and base arrays)
        print("Preparing dataset...")
        cat_dims, cat_idxs, con_idxs, X_tr0, y_tr0, X_v0, y_v0, X_te0, y_te0, train_mean0, train_std0 = \
            data_prep(df, int(self.opt.dset_seed), "binary", datasplit=[0.65, 0.15, 0.20])

        # store for inference consistency
        self._train_cat_idxs = list(cat_idxs)
        self._train_con_idxs = list(con_idxs)

        # CV mode
        if getattr(self.opt, "cv", False):
            print("===== 5-FOLD Stratified CV begins =====")

            X_all_data = np.concatenate([X_tr0["data"], X_v0["data"], X_te0["data"]], axis=0)
            X_all_mask = np.concatenate([X_tr0["mask"], X_v0["mask"], X_te0["mask"]], axis=0)
            y_all = np.concatenate([_y_from_split(y_tr0), _y_from_split(y_v0), _y_from_split(y_te0)], axis=0).astype(np.int64)
            assert len(np.unique(y_all)) >= 2, "y_all must contain at least two classes for StratifiedKFold."

            def pack(X_data, X_mask, idx):
                return {"data": X_data[idx], "mask": X_mask[idx]}

            # Save initial model weights for fold reset
            self.saint.to(self.device)
            init_state = copy.deepcopy(self.saint.state_dict())

            patience = int(getattr(self.opt, "early_patience", 50))
            inner_valid_ratio = float(getattr(self.opt, "valid_ratio", 0.15))
            batch_size = int(self.opt.batchsize)

            folds_metrics = []
            all_predictions = []

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(self.opt.dset_seed))

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all_data, y_all), start=1):
                print(f"\n---- Fold {fold_idx}/5 ----")
                fold_seed = int(getattr(self.opt, "set_seed", self.opt.dset_seed)) + fold_idx
                set_seed(fold_seed)

                X_train_full_data = X_all_data[train_idx]
                X_train_full_mask = X_all_mask[train_idx]
                y_train_full = y_all[train_idx]

                X_test = pack(X_all_data, X_all_mask, test_idx)
                y_test_vec = y_all[test_idx]
                Y_test_dict = {"data": y_test_vec.reshape(-1, 1)}

                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=inner_valid_ratio,
                    random_state=int(self.opt.dset_seed) + fold_idx,
                )
                tr_sub_idx, va_sub_idx = next(sss.split(X_train_full_data, y_train_full))

                X_train = pack(X_train_full_data, X_train_full_mask, tr_sub_idx)
                y_train_vec = y_train_full[tr_sub_idx]
                Y_train_dict = {"data": y_train_vec.reshape(-1, 1)}

                X_valid = pack(X_train_full_data, X_train_full_mask, va_sub_idx)
                y_valid_vec = y_train_full[va_sub_idx]
                Y_valid_dict = {"data": y_valid_vec.reshape(-1, 1)}

                # fold-specific mean/std for continuous features
                con_idxs_arr = np.array(con_idxs, dtype=int) if len(con_idxs) > 0 else np.array([], dtype=int)
                if con_idxs_arr.size > 0:
                    tr_con = X_train["data"][:, con_idxs_arr].astype(np.float32)
                    tr_mean = tr_con.mean(axis=0)
                    tr_std = tr_con.std(axis=0)
                    tr_std[tr_std == 0] = 1.0
                else:
                    tr_mean = np.array([], dtype=np.float32)
                    tr_std = np.array([], dtype=np.float32)

                continuous_mean_std = np.array([tr_mean, tr_std]).astype(np.float32)
                self.opt.continuous_mean_std = continuous_mean_std

                train_ds = DataSetCatCon(X_train, Y_train_dict, cat_idxs, continuous_mean_std)
                valid_ds = DataSetCatCon(X_valid, Y_valid_dict, cat_idxs, continuous_mean_std)
                test_ds  = DataSetCatCon(X_test,  Y_test_dict,  cat_idxs, continuous_mean_std)

                imbalance_mode = getattr(self.opt, "imbalance_mode", "none")
                y_train_np = y_train_vec.astype(int)
                pos_count = int((y_train_np == 1).sum())
                neg_count = int((y_train_np == 0).sum())

                # DataLoader
                if imbalance_mode == "oversample" and pos_count > 0 and neg_count > 0:
                    class_counts = np.bincount(y_train_np, minlength=2).astype(float)
                    class_weights = 1.0 / np.maximum(class_counts, 1.0)
                    samples_weight = torch.from_numpy(class_weights[y_train_np]).float()
                    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
                    trainloader = DataLoader(
                        train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                        num_workers=8, collate_fn=_collate_to_tensors
                    )
                else:
                    trainloader = DataLoader(
                        train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=8, collate_fn=_collate_to_tensors
                    )

                validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=_collate_to_tensors)
                testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=_collate_to_tensors)

                # Reset model to initial state
                self.saint.load_state_dict(init_state)
                self.saint.vision_dset = vision_dset
                self.saint.to(self.device)

                # Optional pretraining
                if getattr(self.opt, "pretrain", False):
                    from pretraining import SAINT_pretrain
                    self.saint = SAINT_pretrain(
                        self.saint, cat_idxs, X_train, Y_train_dict,
                        continuous_mean_std, self.opt, self.device
                    )

                # Loss
                pos_weight_tensor = None
                if imbalance_mode in ["class_weight", "focal"] and pos_count > 0 and neg_count > 0:
                    pw = float(neg_count) / float(pos_count)
                    pos_weight_tensor = torch.tensor(pw, device=self.device)

                if imbalance_mode == "class_weight":
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(self.device)
                else:
                    criterion = nn.BCEWithLogitsLoss().to(self.device)

                focal_alpha = getattr(self.opt, "focal_alpha", 0.25)
                focal_gamma = getattr(self.opt, "focal_gamma", 2.0)

                def focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma):
                    targets = targets.float()
                    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                    p = torch.sigmoid(logits)
                    pt = p * targets + (1.0 - p) * (1.0 - targets)
                    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets) if alpha is not None else 1.0
                    weight = (1.0 - pt).pow(gamma) * alpha_t
                    return (bce * weight).mean()

                # Optimizer
                opt_name = getattr(self.opt, "optimizer", "AdamW")
                if opt_name == "SGD":
                    optimizer = optim.SGD(self.saint.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)
                    from .utils import get_scheduler
                    scheduler = get_scheduler(self.opt, optimizer)
                elif opt_name == "Adam":
                    optimizer = optim.Adam(self.saint.parameters(), lr=self.opt.lr)
                    scheduler = None
                elif opt_name == "AdamW":
                    optimizer = optim.AdamW(self.saint.parameters(), lr=self.opt.lr)
                    scheduler = None
                else:
                    raise ValueError(f"Unknown optimizer: {opt_name}")

                # epoch-wise loss evaluators (train/valid)
                def _epoch_loss(loader):
                    self.saint.eval()
                    losses = []
                    with torch.no_grad():
                        for data in loader:
                            x_categ, x_cont, y_gts, cat_mask, con_mask = (t.to(self.device) for t in data)
                            _, x_categ_enc, x_cont_enc = embed_data_mask(
                                x_categ, x_cont, cat_mask, con_mask, self.saint, vision_dset
                            )
                            reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                            y_reps = reps[:, 0, :]
                            logits = self.saint.mlpfory(y_reps).squeeze(-1)
                            targets = y_gts.float().squeeze()
                            loss = focal_loss(logits, targets) if imbalance_mode == "focal" else criterion(logits, targets)
                            losses.append(loss.detach().cpu().item())
                    return float(np.mean(losses)) if len(losses) > 0 else float("nan")

                def _collect_logits_and_labels(loader):
                    self.saint.eval()
                    all_logits, all_y = [], []
                    with torch.no_grad():
                        for data in loader:
                            x_categ, x_cont, y_gts, cat_mask, con_mask = (t.to(self.device) for t in data)
                            _, x_categ_enc, x_cont_enc = embed_data_mask(
                                x_categ, x_cont, cat_mask, con_mask, self.saint, vision_dset
                            )
                            reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                            y_reps = reps[:, 0, :]
                            logits = self.saint.mlpfory(y_reps).squeeze(-1)
                            all_logits.append(logits.detach().cpu())
                            all_y.append(y_gts.detach().cpu().float().squeeze())
                    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_y, dim=0).numpy()

                def _fit_platt_scaling(valid_logits_np, valid_y_np, max_iter=50):
                    y_np = np.asarray(valid_y_np).reshape(-1)
                    if len(np.unique(y_np)) < 2:
                        return None, None

                    z = torch.tensor(valid_logits_np, dtype=torch.float32, device=self.device)
                    y = torch.tensor(valid_y_np, dtype=torch.float32, device=self.device)

                    a = torch.nn.Parameter(torch.tensor(1.0, device=self.device))
                    b = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    opt_lbfgs = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

                    def closure():
                        opt_lbfgs.zero_grad()
                        loss = loss_fn(a * z + b, y)
                        loss.backward()
                        return loss

                    opt_lbfgs.step(closure)
                    return float(a.detach().cpu().item()), float(b.detach().cpu().item())

                # Train
                fold_dir = os.path.join(base_dir, f"fold{fold_idx}")
                os.makedirs(fold_dir, exist_ok=True)

                best_model_ckpt = os.path.join(fold_dir, "best_model.pt")   # for training reload
                best_infer_ckpt = os.path.join(fold_dir, "best_infer.pt")   # final payload

                best_valid_loss = float("inf")
                wait = 0
                best_epoch = -1

                print("Training begins now.")
                for epoch in range(int(self.opt.epochs)):
                    self.saint.train()
                    tr_losses = []

                    for data in trainloader:
                        optimizer.zero_grad()
                        x_categ, x_cont, y_gts, cat_mask, con_mask = (t.to(self.device) for t in data)
                        _, x_categ_enc, x_cont_enc = embed_data_mask(
                            x_categ, x_cont, cat_mask, con_mask, self.saint, vision_dset
                        )
                        reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                        y_reps = reps[:, 0, :]
                        logits = self.saint.mlpfory(y_reps).squeeze(-1)
                        targets = y_gts.float().squeeze()

                        loss = focal_loss(logits, targets) if imbalance_mode == "focal" else criterion(logits, targets)
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                        tr_losses.append(loss.detach().cpu().item())

                    train_loss = float(np.mean(tr_losses)) if len(tr_losses) > 0 else float("nan")
                    valid_loss = _epoch_loss(validloader)

                    # per-epoch validation loss log
                    print(f"[FOLD {fold_idx} | EPOCH {epoch+1}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")

                    # early stopping: validation loss minimize
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        wait = 0
                        best_epoch = epoch + 1
                        torch.save({"model": self.saint.state_dict(), "opt": self.opt}, best_model_ckpt)
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"[FOLD {fold_idx}] Early stopping at epoch {epoch+1} (best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f})")
                            break

                # Load best model (loss-min)
                ckpt = torch.load(best_model_ckpt, map_location=self.device)
                self.saint.load_state_dict(ckpt["model"])
                self.saint.eval()

                # Calibration + thresholding (as-is)
                v_logits_np, yv_np = _collect_logits_and_labels(validloader)
                t_logits_np, yt_np = _collect_logits_and_labels(testloader)

                pv_raw = _sigmoid_np(v_logits_np)
                pt_raw = _sigmoid_np(t_logits_np)

                calib_a, calib_b = _fit_platt_scaling(v_logits_np, yv_np, max_iter=50)
                if (calib_a is None) or (calib_b is None):
                    pv_cal, pt_cal = pv_raw, pt_raw
                    self.calib_a, self.calib_b = None, None
                else:
                    pv_cal = _sigmoid_np(calib_a * v_logits_np + calib_b)
                    pt_cal = _sigmoid_np(calib_a * t_logits_np + calib_b)
                    self.calib_a = torch.tensor(calib_a, dtype=torch.float32, device=self.device)
                    self.calib_b = torch.tensor(calib_b, dtype=torch.float32, device=self.device)

                # Threshold from calibrated valid (maximize F1)
                if len(np.unique(yv_np)) < 2:
                    best_thr = 0.5
                else:
                    prec, rec, thr = precision_recall_curve(yv_np, pv_cal)
                    f1s = 2 * prec * rec / (prec + rec + 1e-12)
                    f1s = np.nan_to_num(f1s, nan=0.0, posinf=0.0, neginf=0.0)
                    best_thr = 0.5 if len(thr) == 0 else float(thr[int(np.argmax(f1s[:-1]))])

                y_pred = (pt_cal >= best_thr).astype(np.int64)
                valid_pred = (pv_cal >= best_thr).astype(np.int64)

                from .utils import count_parameters
                total_parameters = count_parameters(self.saint)

                fold_metrics = {
                    "fold": fold_idx,
                    "params": int(total_parameters),
                    "best_epoch_earlystop": int(best_epoch),
                    "best_valid_loss_earlystop": float(best_valid_loss),

                    "valid_acc": float(accuracy_score(yv_np, valid_pred)),
                    "valid_auroc": _safe_auroc(yv_np, pv_cal),
                    "valid_auprc": float(average_precision_score(yv_np, pv_cal)),

                    "calib_a": float(calib_a) if calib_a is not None else None,
                    "calib_b": float(calib_b) if calib_b is not None else None,
                    "best_threshold": float(best_thr),

                    "test_acc": float(accuracy_score(yt_np, y_pred)),
                    "test_auroc": _safe_auroc(yt_np, pt_cal),
                    "test_auprc": float(average_precision_score(yt_np, pt_cal)),
                    "test_f1": float(f1_score(yt_np, y_pred)),
                }
                folds_metrics.append(fold_metrics)

                all_predictions.append({
                    "fold": fold_idx,
                    "y_true": yt_np.tolist(),
                    "y_score_raw": pt_raw.tolist(),
                    "y_prob_calibrated": pt_cal.tolist(),
                    "y_pred": y_pred.tolist(),
                    "threshold": float(best_thr),
                    "calib_a": fold_metrics["calib_a"],
                    "calib_b": fold_metrics["calib_b"],
                })

                with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                    json.dump(fold_metrics, f, indent=2)
                with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
                    json.dump(all_predictions[-1], f, indent=2)
                print(f"[FOLD {fold_idx}] Saved metrics & predictions")

                # Save inference payload
                payload = self._pack_infer_ckpt(
                    model_state=self.saint.state_dict(),
                    continuous_mean_std=continuous_mean_std,
                    best_threshold=best_thr,
                    cat_idxs=cat_idxs,
                    con_idxs=con_idxs,
                    calib_a=calib_a,
                    calib_b=calib_b
                )
                torch.save(payload, best_infer_ckpt)
                print(f"[FOLD {fold_idx}] Saved inference ckpt: {best_infer_ckpt}")

            # CV summary
            dfm = pd.DataFrame(folds_metrics)
            dfm.to_csv(os.path.join(base_dir, "cv_metrics_summary.tsv"), sep="\t", index=False)

            summary = {
                "n_folds": 5,
                "pathway": self.pathway,
                "metrics_mean": dfm[["test_acc", "test_auroc", "test_auprc", "test_f1"]].mean().to_dict(),
                "metrics_std": dfm[["test_acc", "test_auroc", "test_auprc", "test_f1"]].std(ddof=1).to_dict(),
            }
            with open(os.path.join(base_dir, "cv_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print("\n===== 5-FOLD CV finished =====")
            print("Per-fold metrics:\n", dfm)
            print("CV mean:", summary["metrics_mean"])
            print("CV std :", summary["metrics_std"])

            # Keep last fold outputs
            self.best_test_y = torch.tensor(all_predictions[-1]["y_true"])
            self.best_pred_y = torch.tensor(all_predictions[-1]["y_pred"])
            self.best_threshold = float(all_predictions[-1]["threshold"])

        # Single split mode
        else:
            val_losses = []
            val_auprcs = []
            early_stop_epoch = int(self.opt.epochs)

            continuous_mean_std = np.array([train_mean0, train_std0]).astype(np.float32)
            self.opt.continuous_mean_std = continuous_mean_std

            train_ds = DataSetCatCon(X_tr0, y_tr0, cat_idxs, continuous_mean_std)
            valid_ds = DataSetCatCon(X_v0,  y_v0,  cat_idxs, continuous_mean_std)
            test_ds  = DataSetCatCon(X_te0, y_te0, cat_idxs, continuous_mean_std)

            imbalance_mode = getattr(self.opt, "imbalance_mode", "none")
            batch_size = int(self.opt.batchsize)

            y_train_vec = _y_from_split(y_tr0)
            y_train_np = y_train_vec.astype(int)
            pos_count = int((y_train_np == 1).sum())
            neg_count = int((y_train_np == 0).sum())

            if imbalance_mode == "oversample" and pos_count > 0 and neg_count > 0:
                class_counts = np.bincount(y_train_np, minlength=2).astype(float)
                class_weights = 1.0 / np.maximum(class_counts, 1.0)
                samples_weight = torch.from_numpy(class_weights[y_train_np]).float()
                sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
                trainloader = DataLoader(
                    train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                    num_workers=8, collate_fn=_collate_to_tensors
                )
            else:
                trainloader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=8, collate_fn=_collate_to_tensors
                )

            validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=_collate_to_tensors)
            testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=_collate_to_tensors)

            self.saint.to(self.device)

            if getattr(self.opt, "pretrain", False):
                from pretraining import SAINT_pretrain
                self.saint = SAINT_pretrain(self.saint, cat_idxs, X_tr0, y_tr0, continuous_mean_std, self.opt, self.device)

            pos_weight_tensor = None
            if imbalance_mode in ["class_weight", "focal"] and pos_count > 0 and neg_count > 0:
                pw = float(neg_count) / float(pos_count)
                pos_weight_tensor = torch.tensor(pw, device=self.device)

            if imbalance_mode == "class_weight":
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(self.device)
            else:
                criterion = nn.BCEWithLogitsLoss().to(self.device)

            focal_alpha = getattr(self.opt, "focal_alpha", 0.25)
            focal_gamma = getattr(self.opt, "focal_gamma", 2.0)

            def focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma):
                targets = targets.float()
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                p = torch.sigmoid(logits)
                pt = p * targets + (1.0 - p) * (1.0 - targets)
                alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets) if alpha is not None else 1.0
                weight = (1.0 - pt).pow(gamma) * alpha_t
                return (bce * weight).mean()

            opt_name = getattr(self.opt, "optimizer", "AdamW")
            if opt_name == "SGD":
                optimizer = optim.SGD(self.saint.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=5e-4)
                from .utils import get_scheduler
                scheduler = get_scheduler(self.opt, optimizer)
            elif opt_name == "Adam":
                optimizer = optim.Adam(self.saint.parameters(), lr=self.opt.lr)
                scheduler = None
            elif opt_name == "AdamW":
                optimizer = optim.AdamW(self.saint.parameters(), lr=self.opt.lr)
                scheduler = None
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")

            def _epoch_loss(loader):
                self.saint.eval()
                losses = []
                with torch.no_grad():
                    for data in loader:
                        x_categ, x_cont, y_gts, cat_mask, con_mask = (t.to(self.device) for t in data)
                        _, x_categ_enc, x_cont_enc = embed_data_mask(
                            x_categ, x_cont, cat_mask, con_mask, self.saint, vision_dset
                        )
                        reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                        y_reps = reps[:, 0, :]
                        logits = self.saint.mlpfory(y_reps).squeeze(-1)
                        targets = y_gts.float().squeeze()
                        loss = focal_loss(logits, targets) if imbalance_mode == "focal" else criterion(logits, targets)
                        losses.append(loss.detach().cpu().item())
                return float(np.mean(losses)) if len(losses) > 0 else float("nan")

            single_dir = os.path.join(base_dir, "single")
            os.makedirs(single_dir, exist_ok=True)
            best_model_ckpt = os.path.join(single_dir, "best_model.pt")
            best_infer_ckpt = os.path.join(single_dir, "best_infer.pt")

            patience = int(getattr(self.opt, "early_patience", 30))
            best_valid_loss = float("inf")
            wait = 0
            best_epoch = -1

            print("Training begins now.")
            for epoch in range(int(self.opt.epochs)):
                self.saint.train()
                tr_losses = []

                for data in trainloader:
                    optimizer.zero_grad()
                    x_categ, x_cont, y_gts, cat_mask, con_mask = (t.to(self.device) for t in data)
                    _, x_categ_enc, x_cont_enc = embed_data_mask(
                        x_categ, x_cont, cat_mask, con_mask, self.saint, vision_dset
                    )
                    reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    logits = self.saint.mlpfory(y_reps).squeeze(-1)
                    targets = y_gts.float().squeeze()

                    loss = focal_loss(logits, targets) if imbalance_mode == "focal" else criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    tr_losses.append(loss.detach().cpu().item())

                train_loss = float(np.mean(tr_losses)) if len(tr_losses) > 0 else float("nan")
                valid_loss = _epoch_loss(validloader)

                # per-epoch validation loss log
                print(f"[SINGLE | EPOCH {epoch+1}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
                val_losses.append(valid_loss)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    wait = 0
                    best_epoch = epoch + 1
                    torch.save({"model": self.saint.state_dict(), "opt": self.opt}, best_model_ckpt)
                else:
                    wait += 1
                    if wait >= patience:
                        early_stop_epoch = epoch + 1
                        print(f"[SINGLE] Early stopping at epoch {epoch+1} (best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f})")
                        break

            ckpt = torch.load(best_model_ckpt, map_location=self.device)
            self.saint.load_state_dict(ckpt["model"])
            self.saint.eval()

            with torch.no_grad():
                from .utils import classification_scores
                v_acc, v_auc, yv, pv = classification_scores(
                    self.saint, validloader, self.device, "binary", vision_dset,
                    threshold=0.5, augment=False
                )
                t_acc, t_auc, yt, pt = classification_scores(
                    self.saint, testloader, self.device, "binary", vision_dset,
                    threshold=0.5, augment=False
                )

            # Threshold from valid (maximize F1) on prob
            yv_np = yv.numpy()
            pv_np = pv.numpy()
            if len(np.unique(yv_np)) < 2:
                best_thr = 0.5
            else:
                from sklearn.metrics import precision_recall_curve
                prec, rec, thr = precision_recall_curve(yv_np, pv_np)
                f1s = 2 * prec * rec / (prec + rec + 1e-12)
                f1s = np.nan_to_num(f1s, nan=0.0, posinf=0.0, neginf=0.0)
                best_thr = 0.5 if len(thr) == 0 else float(thr[int(np.argmax(f1s[:-1]))])

            payload = self._pack_infer_ckpt(
                model_state=self.saint.state_dict(),
                continuous_mean_std=continuous_mean_std,
                best_threshold=best_thr,
                cat_idxs=cat_idxs,
                con_idxs=con_idxs,
                calib_a=None,
                calib_b=None
            )
            torch.save(payload, best_infer_ckpt)
            print(f"[SINGLE] Saved inference ckpt: {best_infer_ckpt}")

            yt_np = yt.numpy().astype(int)
            pt_np = pt.numpy().astype(float)

            y_pred = (pt_np >= best_thr).astype(np.int64)

            test_acc = float(accuracy_score(yt_np, y_pred))
            test_auroc = _safe_auroc(yt_np, pt_np)
            test_auprc = float(average_precision_score(yt_np, pt_np))
            test_f1 = float(f1_score(yt_np, y_pred))

            from .utils import count_parameters
            total_parameters = count_parameters(self.saint)
            from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
            print(f"TOTAL NUMBER OF SAINT PARAMS: {int(total_parameters)}")
            print(f"Accuracy on best model:  {accuracy_score(yt.numpy(), y_pred):.3f}")
            print(f"AUROC on best model:     {_safe_auroc(yt.numpy(), pt.numpy()):.3f}")
            print(f"AUPRC on best model:     {average_precision_score(yt.numpy(), pt.numpy()):.3f}")
            print(f"F1 on best model:        {f1_score(yt.numpy(), y_pred):.3f}")
            print(f"Chosen threshold:        {best_thr:.3f}")

            self.best_test_y = yt
            self.best_pred_y = torch.from_numpy(y_pred)
            self.best_threshold = float(best_thr)

            single_metrics = {
                "mode": "single",
                "pathway": self.pathway,
                "best_epoch": int(best_epoch),
                "early_stop_epoch": int(early_stop_epoch),
                "val_loss_by_epoch": val_losses,
                "val_auprc_by_epoch": val_auprcs,
                "test_acc": test_acc,
                "test_auroc": test_auroc,
                "test_auprc": test_auprc,
                "test_f1": test_f1,
                "best_threshold": float(best_thr),
            }
            with open(os.path.join(single_dir, "single_metrics.json"), "w") as f:
                json.dump(single_metrics, f, indent=2)

    # ---------- inference prep ----------
    def _prepare_inputs(self, X):
        self.saint.to(self.device)
        from .data_preprocess import data_prep_forward

        n_cont = int(getattr(self.saint, "num_continuous", 256))
        _, cat_idxs_fwd, con_idxs_fwd, X_prepared = data_prep_forward(X, n_cont)

        X_mask = X_prepared["mask"]
        X_data = X_prepared["data"]

        cat_cols = list(self._train_cat_idxs) if self._train_cat_idxs is not None else list(cat_idxs_fwd)
        con_cols = list(self._train_con_idxs) if self._train_con_idxs is not None else list(con_idxs_fwd)

        if len(cat_cols) > 0:
            X1 = torch.as_tensor(X_data[:, cat_cols], dtype=torch.long, device=self.device)
            X1_mask = torch.as_tensor(X_mask[:, cat_cols], dtype=torch.long, device=self.device)
        else:
            X1 = torch.empty((X_data.shape[0], 0), dtype=torch.long, device=self.device)
            X1_mask = torch.empty((X_data.shape[0], 0), dtype=torch.long, device=self.device)

        X2 = torch.as_tensor(X_data[:, con_cols], dtype=torch.float32, device=self.device)
        X2_mask = torch.as_tensor(X_mask[:, con_cols], dtype=torch.long, device=self.device)

        B = X2.shape[0]
        cls = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        cls_mask = torch.ones((B, 1), dtype=torch.long, device=self.device)

        mean, std = self.opt.continuous_mean_std
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        std = torch.where(std == 0, torch.ones_like(std), std)

        if mean.numel() != X2.shape[1]:
            raise RuntimeError(f"continuous_mean_std size mismatch: mean({mean.numel()}) vs X2({X2.shape[1]})")

        X2 = (X2 - mean) / std

        x_categ = torch.cat((cls, X1), dim=1)
        cat_mask = torch.cat((cls_mask, X1_mask), dim=1)

        x_cont = X2
        con_mask = X2_mask

        return x_categ, x_cont, cat_mask, con_mask

    def _logits_from_prepared(self, x_categ, x_cont, cat_mask, con_mask):
        from .augmentations import embed_data_mask

        _, x_categ_enc, x_cont_enc = embed_data_mask(
            x_categ, x_cont, cat_mask, con_mask,
            self.saint, self.opt.vision_dset
        )

        reps = self.saint.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        logits = self.saint.mlpfory(y_reps).squeeze(-1)
        return logits

    def forward(self, X):
        x_categ, x_cont, cat_mask, con_mask = self._prepare_inputs(X)
        logits = self._logits_from_prepared(x_categ, x_cont, cat_mask, con_mask)
        p_hat = self._apply_calibration(logits).unsqueeze(1).float().cpu()
        torch.cuda.empty_cache()
        return p_hat

    def predict(self, X):
        x_categ, x_cont, cat_mask, con_mask = self._prepare_inputs(X)
        logits = self._logits_from_prepared(x_categ, x_cont, cat_mask, con_mask)
        thr = float(getattr(self, "best_threshold", 0.5))
        y_pred = (self._apply_calibration(logits) >= thr).float().unsqueeze(1)
        torch.cuda.empty_cache()
        return y_pred


def train_autoencoder(
    model: Autoencoder,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    l1_alpha: float = 1e-4,
    save_path: Optional[str] = None,
    seed: int = 1,
    verbose: bool = True,
):
    def fix_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    fix_seed(seed)
    model = model.to(device)

    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    Xva = torch.as_tensor(X_valid, dtype=torch.float32)

    tr_loader = DataLoader(Xtr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    va_loader = DataLoader(Xva, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    optim_ = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss(reduction="mean")

    best_val = float("inf")
    wait = 0
    best_epoch = -1
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        tr_losses = []
        for xb in tr_loader:
            xb = xb.to(device)
            optim_.zero_grad()
            z, xhat = model(xb)

            recon = mse(xhat, xb)
            l1 = z.abs().mean()  # stable scale
            loss = recon + l1_alpha * l1

            loss.backward()
            optim_.step()
            tr_losses.append(loss.detach().cpu().item())

        model.eval()
        va_mses = []
        with torch.no_grad():
            for xb in va_loader:
                xb = xb.to(device)
                z, xhat = model(xb)
                va_mses.append(mse(xhat, xb).detach().cpu().item())
        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        valid_mse = float(np.mean(va_mses)) if va_mses else float("nan")

        if verbose:
            print(f"[AE | EPOCH {epoch+1}] train_loss={train_loss:.6f} valid_mse={valid_mse:.6f}")

        if valid_mse < best_val:
            best_val = valid_mse
            best_epoch = epoch + 1
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"model": best_state, "best_epoch": best_epoch, "best_valid_mse": best_val}, save_path)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[AE] Early stopping at epoch {epoch+1} (best_epoch={best_epoch}, best_valid_mse={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_epoch": best_epoch, "best_valid_mse": best_val}