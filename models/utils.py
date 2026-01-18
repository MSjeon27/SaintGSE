import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score, precision_score, recall_score, f1_score
import numpy as np
from .augmentations import embed_data_mask
import torch.nn as nn

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model, dloader, device):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob   = torch.empty(0).to(device)
    with torch.no_grad():
        for data in dloader:
            x_categ, x_cont, cat_mask, con_mask = (t.to(device) for t in data)
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, getattr(model, 'vision_dset', None)
            )
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, model.num_categories-1, :]
            outs = model.mlpfory(y_reps)  # (B,1) or (B,C)
            if outs.shape[-1] == 1:
                logits = outs.squeeze(-1)
                p = torch.sigmoid(logits)
                pred = (p >= 0.5).float()
            else:
                p_full = torch.softmax(outs, dim=1)
                p = p_full[:, -1]
                pred = torch.argmax(outs, dim=1).float()
            y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
            y_pred = torch.cat([y_pred, pred], dim=0)
            prob   = torch.cat([prob, p], dim=0)
    acc = (y_pred == y_test).float().mean()*100
    auc = roc_auc_score(y_true=y_test.cpu(), y_score=prob.cpu())
    return acc, auc


def classification_scores(model, dloader, device, task, vision_dset,
                          threshold: float = 0.5, augment: bool = False):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    if augment:
        from .augmentations import embed_data_mask as _embed
    else:
        from .augmentations import embed_data_nomask as _embed

    model.eval()

    y_true_list, y_pred_list, proba_list = [], [], []
    with torch.no_grad():
        for data in dloader:
            x_categ = data[0].to(device)
            x_cont  = data[1].to(device)
            y_gts   = data[2].to(device)

            if augment:
                cat_mask= data[3].to(device)
                con_mask= data[4].to(device)
                _, x_categ_enc, x_cont_enc = _embed(
                    x_categ, x_cont, cat_mask, con_mask, model, vision_dset
                )
            else:
                _, x_categ_enc, x_cont_enc = _embed(
                    x_categ, x_cont, model, vision_dset
                )

            reps   = model.transformer(x_categ_enc, x_cont_enc) # (B, N, D)
            y_reps = reps[:, 0, :]  # (B, D)
            outs   = model.mlpfory(y_reps)  # (B, 1) or (B, C)

            if outs.shape[-1] == 1:
                # Binary: BCE logits
                logits = outs.squeeze(-1)   # (B,)
                proba  = torch.sigmoid(logits)  # (B,)
                preds  = (proba >= threshold).float()   # (B,)
            else:
                # Multi-class: CE logits
                probs_full = torch.softmax(outs, dim=1) # (B, C)
                proba = probs_full[:, 1]
                preds = torch.argmax(outs, dim=1).float()

            y_true_list.append(y_gts.squeeze().float().detach())
            y_pred_list.append(preds.detach())
            proba_list.append(proba.detach())

    y_true = torch.cat(y_true_list).cpu()
    y_pred = torch.cat(y_pred_list).cpu()
    prob   = torch.cat(proba_list).cpu()

    acc_percent = (y_pred == y_true).sum().float() / y_true.shape[0] * 100.0

    try:
        auc = roc_auc_score(y_true=y_true.numpy(), y_score=prob.numpy())
    except Exception:
        auc = float('nan')

    return acc_percent.item(), auc, y_true.cpu(), prob.cpu()

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred, y_outs.squeeze(-1)], dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse
