import torch
import numpy as np


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device

    # categorical → embedding index offset
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ) # (B, n_cat, dim)

    # continuous → per-feature MLP
    B, C = x_cont.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(B, C, model.dim, device=device)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    elif model.cont_embeddings == 'pos_singleMLP':
        shared_mlp = model.simple_MLP[0]
        x_flat = x_cont.reshape(-1).unsqueeze(1)    # (B*C, 1)
        y_flat = shared_mlp(x_flat) # (B*C, dim)
        x_cont_enc = y_flat.view(B, C, model.dim).to(device)
    else:
        raise ValueError("cont_embeddings must be 'MLP' or 'pos_singleMLP'")

    # masking (0 = mask, 1 = keep)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    # optional positional encodings
    if vision_dset:
        pos = torch.arange(x_categ.shape[-1], device=device).repeat(x_categ.shape[0], 1)
        pos_enc = model.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc


def embed_data_nomask(x_categ, x_cont, model, vision_dset=False):
    """
    Deterministic embedding without using any masks.
    x_categ: (B, n_cat)
    x_cont : (B, n_cont)
    """
    device = x_cont.device

    # categorical
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)

    # continuous
    B, C = x_cont.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(B, C, model.dim, device=device)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    elif model.cont_embeddings == 'pos_singleMLP':
        shared_mlp = model.simple_MLP[0]
        x_flat = x_cont.reshape(-1).unsqueeze(1)
        y_flat = shared_mlp(x_flat)
        x_cont_enc = y_flat.view(B, C, model.dim).to(device)
    else:
        raise ValueError("cont_embeddings must be 'MLP' or 'pos_singleMLP'")

    if vision_dset:
        pos = torch.arange(x_categ.shape[-1], device=device).repeat(x_categ.shape[0], 1)
        pos_enc = model.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc


def mixup_data(x1, x2, lam=1.0, y=None, use_cuda=True):
    """
    Feature-level mixup.
    x1, x2: (B, D...)
    lam   : mixup lambda
    y     : (B,) or (B,1), optional
    """
    batch_size = x1.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]

    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b

    return mixed_x1, mixed_x2


def add_noise(x_categ, x_cont, noise_params={'noise_type': ['cutmix'], 'lambda': 0.1}):
    """
    Simple feature-level noise:
      - 'cutmix': replace part of features with another sample
      - 'missing': mask out part of features
    """
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size(0)

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2, x_categ.shape, p=[lam, 1 - lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2, x_cont.shape, p=[lam, 1 - lam])).to(device)
        x1, x2 = x_categ[index, :], x_cont[index, :]
        x_categ_corr, x_cont_corr = x_categ.clone().detach(), x_cont.clone().detach()
        x_categ_corr[cat_corr == 0] = x1[cat_corr == 0]
        x_cont_corr[con_corr == 0] = x2[con_corr == 0]
        return x_categ_corr, x_cont_corr

    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2, x_categ.shape, p=[lam, 1 - lam])
        x_cont_mask = np.random.choice(2, x_cont.shape, p=[lam, 1 - lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ, x_categ_mask), torch.mul(x_cont, x_cont_mask)

    else:
        print("yet to write this")
        return x_categ, x_cont
