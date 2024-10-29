import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from .model import *


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep(df, seed, task, datasplit=[.65, .15, .2]):
    
    np.random.seed(seed) 
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = [X.columns.get_loc(col) for col in categorical_columns]
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    
    for col in cont_columns:
        X[col].fillna(X.loc[train_indices, col].mean(), inplace=True)
    
    y = y.values
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(y)
    
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std



def data_prep_forward(tensor, num_features):
    """
    Prepare data for prediction by extracting necessary features and applying necessary transformations.
    Args:
        tensor (Tensor): The input tensor with shape [batch_size, num_features].
        num_features (int): Total number of features in the tensor.
    Returns:
        Tuple: containing prepared categorical and continuous features, and associated metadata.
    """
    # Initialize empty lists for categorical and continuous column indices
    categorical_columns = []
    continuous_columns = list(range(num_features))
    
    # Create masks for missing values
    mask = torch.ones_like(tensor, dtype=torch.float32)
    mask[tensor == -1] = 0  # Mark missing values with 0 in the mask
    
    # If there are categorical columns, perform encoding
    cat_dims = []
    if categorical_columns:
        for col in categorical_columns:
            unique_vals = torch.unique(tensor[:, col])
            cat_dims.append(unique_vals.size(0))
            value_to_int = {val.item(): i for i, val in enumerate(unique_vals)}
            tensor[:, col] = torch.tensor([value_to_int[val.item()] for val in tensor[:, col]], dtype=torch.float32)
    
    X_prepared = {
        'data': tensor,
        'mask': mask
    }
    
    return cat_dims, categorical_columns, continuous_columns, X_prepared


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        self.y = Y['data']#.astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
