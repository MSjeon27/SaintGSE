from .model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

import os

class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
        ):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def preprocess_pred(self, df):
        X = df.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled

class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
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
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

        
    def forward(self, x_categ, x_cont):
        
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:,:self.num_categories,:])
        con_outs = self.mlp2(x[:,self.num_categories:,:])
        return cat_outs, con_outs 
    
class SaintGSE(nn.Module):
    def __init__(
            self,
            device,
            latent_df,
            target_pathway, 
            opt
        ):
        super().__init__()
        self.device = device
        self.latent_df = latent_df
        self.pathway = target_pathway
        self.opt = opt
        re_path = self.pathway.replace(' ', '_')
        self.modelsave_path = os.path.join(opt.current_dir, 'datasets', fr'bestmodels/binary/{re_path}/testrun')
        categories_init = np.array([1]).astype(int)
        num_continuous_feature = latent_df.shape[-1] - 1 # The number of continuous features except label (y)

        ##### Setting some hyperparams based on inputs and dataset
        if num_continuous_feature > 100:
            self.opt.embedding_size = min(8,self.opt.embedding_size)
            self.opt.batchsize = min(64, self.opt.batchsize)
        if self.opt.attentiontype != 'col':
            self.opt.transformer_depth = 1
            self.opt.attention_heads = min(4,self.opt.attention_heads)
            self.opt.attention_dropout = 0.8
            self.opt.embedding_size = min(32,self.opt.embedding_size)
            self.opt.ff_dropout = 0.8
            
        self.saint = SAINT(
            categories=categories_init,  # CLS token
            num_continuous=num_continuous_feature, 
            dim=self.opt.embedding_size,
            dim_out=1,
            depth=self.opt.transformer_depth,
            heads=self.opt.attention_heads,
            attn_dropout=self.opt.attention_dropout,
            ff_dropout=self.opt.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=self.opt.cont_embeddings,
            attentiontype=self.opt.attentiontype,
            final_mlp_style=self.opt.final_mlp_style,
            y_dim=2 # number of classes
        )

    def run(self):
        self.latent_df[self.pathway] = self.latent_df['Enrichment Results'].apply(lambda x: 1 if self.pathway in x else 0)
        self.latent_df.drop(columns=['Enrichment Results'], inplace=True)


        # Autoencoder training
        print(f'SaintGSE for designated {self.pathway} started!')
        print(self.latent_df)
        print(f'latent_df shape: {self.latent_df.shape}')

        # Run SaintGSE for designated pathways
        print('Processing the dataset, it might take some time.')
        from .data_preprocess import data_prep
        cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(self.latent_df, self.opt.dset_seed,'binary', datasplit=[.65, .15, .2])

        continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)
        self.opt.continuous_mean_std = continuous_mean_std

        from .data_preprocess import DataSetCatCon
        train_ds = DataSetCatCon(X_train, y_train, cat_idxs, continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=self.opt.batchsize, shuffle=True,num_workers=8)

        valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=self.opt.batchsize, shuffle=False,num_workers=8)

        test_ds = DataSetCatCon(X_test, y_test, cat_idxs, continuous_mean_std)
        testloader = DataLoader(test_ds, batch_size=self.opt.batchsize, shuffle=False,num_workers=8)

        vision_dset = self.opt.vision_dset

        # binary classification
        criterion = nn.CrossEntropyLoss().to(self.device)

        self.saint.to(self.device)

        if self.opt.pretrain:
            from pretraining import SAINT_pretrain
            self.saint = SAINT_pretrain(self.saint, cat_idxs,X_train,y_train, continuous_mean_std, self.opt,self.device)

        # Choosing the optimizer

        if self.opt.optimizer == 'SGD':
            optimizer = optim.SGD(self.saint.parameters(), lr=self.opt.lr,
                                momentum=0.9, weight_decay=5e-4)
            from .utils import get_scheduler
            scheduler = get_scheduler(self.opt, optimizer)
        elif self.opt.optimizer == 'Adam':
            optimizer = optim.Adam(self.saint.parameters(),lr=self.opt.lr)
        elif self.opt.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.saint.parameters(),lr=self.opt.lr)
        best_valid_auroc = 0
        best_valid_accuracy = 0
        best_test_auroc = 0
        best_test_accuracy = 0
        
        print('Training begins now.')
        for epoch in range(self.opt.epochs):
            self.saint.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(self.device), data[1].to(self.device),data[2].to(self.device),data[3].to(self.device),data[4].to(self.device)

                # We are converting the data to embeddings in the next step
                from .augmentations import embed_data_mask
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,self.saint,vision_dset)
                reps = self.saint.transformer(x_categ_enc, x_cont_enc)
                # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:,0,:]
                
                y_outs = self.saint.mlpfory(y_reps)

                # Binary classification loss
                loss = criterion(y_outs,y_gts.squeeze()) 
                loss.backward()
                optimizer.step()
                if self.opt.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()

            # print(running_loss)
            self.saint.eval()
            with torch.no_grad():
                from .utils import classification_scores
                accuracy, auroc, test_y, pred_y = classification_scores(self.saint, validloader, self.device, 'binary',vision_dset)
                test_accuracy, test_auroc, test_test_y, test_pred_y = classification_scores(self.saint, testloader, self.device, 'binary',vision_dset)

                if epoch%5==0:
                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        (epoch + 1, test_accuracy,test_auroc ))
                if accuracy >= best_valid_accuracy and auroc > best_valid_auroc and test_auroc > best_test_auroc:
                    best_valid_auroc = auroc
                    best_valid_accuracy = accuracy
                    best_test_auroc = test_auroc
                    best_test_accuracy = test_accuracy
                    best_test_y = test_test_y
                    best_pred_y = test_pred_y
                    self.best_saint = self.saint
                    
            self.saint.train()

        self.best_test_y = best_test_y
        self.best_pred_y = best_pred_y

        from .utils import count_parameters
        total_parameters = count_parameters(self.saint)
        print('TOTAL NUMBER OF SAINT PARAMS: %d' %(total_parameters))
        print('Accuracy on best model:  %.3f' %(best_test_accuracy))
        print('AUROC on best model:  %.3f' %(best_test_auroc))



    def forward(self, X):
        """
        Predict methods for SHAP
        X: Input data
        Returns: predictions
        """
        self.saint.to(self.device)
        from .data_preprocess import data_prep_forward
        
        cat_dims, cat_idxs, con_idxs, X_prepared = data_prep_forward(X, 256)

        continuous_mean_std_trained = self.opt.continuous_mean_std

        # Preprocessing
        cat_cols = list(cat_idxs)
        X_mask = X_prepared['mask']
        X = X_prepared['data']
        con_cols = list(set(range(X.shape[1])) - set(cat_cols))
        X1 = X[:, cat_cols].int()  # categorical columns
        X2 = X[:, con_cols].float()  # numerical columns
        X1_mask = X_mask[:, cat_cols].int()  # categorical columns
        X2_mask = X_mask[:, con_cols].int()  # numerical columns
        cls = torch.zeros((X1.shape[0], 1), dtype=torch.int, device=X2.device)  # binary classification
        cls_mask = torch.ones((X1.shape[0], 1), dtype=torch.int, device=X2.device)
        
        # Normalize continuous columns
        mean, std = continuous_mean_std_trained
        mean = torch.tensor(mean, dtype=torch.float32, device=X2.device)
        std = torch.tensor(std, dtype=torch.float32, device=X2.device)
        std = torch.where(std == 0, torch.tensor(1.0, device=X2.device), std)  # Avoid division by zero
        X2 = (X2 - mean) / std

        # CLS token
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
        
        # Prepare tensors for SAINT model
        x_categ = torch.cat((cls, X1), dim=1).int().to(self.device)  # No gradient required
        x_cont = X2.float().to(self.device)  # Gradient required
        cat_mask = torch.cat((cls_mask, X1_mask), dim=1).int().to(self.device)
        con_mask = X2_mask.long().to(self.device)

        prob = torch.empty(0).to(self.device)

        from .augmentations import embed_data_mask
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.saint, self.opt.vision_dset)
        reps = self.saint.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = self.saint.mlpfory(y_reps)
                

        m = nn.Softmax(dim=1)

        prob = torch.cat([prob,m(y_outs)[:,-1].unsqueeze(1).float()],dim=0).cpu()
        torch.cuda.empty_cache()

        return prob

    def predict(self, X):
        """
        Predict methods for SHAP
        X: Input data
        Returns: predictions
        """
        self.saint.to(self.device)
        from .data_preprocess import data_prep_forward
        
        cat_dims, cat_idxs, con_idxs, X_prepared = data_prep_forward(X, 256)

        continuous_mean_std_trained = self.opt.continuous_mean_std

        # Preprocessing
        cat_cols = list(cat_idxs)
        X_mask = X_prepared['mask']
        X = X_prepared['data']
        con_cols = list(set(range(X.shape[1])) - set(cat_cols))
        X1 = X[:, cat_cols].int()  # categorical columns
        X2 = X[:, con_cols].float()  # numerical columns
        X1_mask = X_mask[:, cat_cols].int()  # categorical columns
        X2_mask = X_mask[:, con_cols].int()  # numerical columns
        cls = torch.zeros((X1.shape[0], 1), dtype=torch.int, device=X2.device)  # binary classification
        cls_mask = torch.ones((X1.shape[0], 1), dtype=torch.int, device=X2.device)
        
        # Normalize continuous columns
        mean, std = continuous_mean_std_trained
        mean = torch.tensor(mean, dtype=torch.float32, device=X2.device)
        std = torch.tensor(std, dtype=torch.float32, device=X2.device)
        std = torch.where(std == 0, torch.tensor(1.0, device=X2.device), std)  # Avoid division by zero
        X2 = (X2 - mean) / std

        # CLS token
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
        
        # Prepare tensors for SAINT model
        x_categ = torch.cat((cls, X1), dim=1).int().to(self.device)  # No gradient required
        x_cont = X2.float().to(self.device)  # Gradient required
        cat_mask = torch.cat((cls_mask, X1_mask), dim=1).int().to(self.device)
        con_mask = X2_mask.long().to(self.device)

        y_pred = torch.empty(0).to(self.device)

        from .augmentations import embed_data_mask
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.saint, self.opt.vision_dset)
        reps = self.saint.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = self.saint.mlpfory(y_reps)
                

        y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).unsqueeze(1).float()], dim=0)

        torch.cuda.empty_cache()

        return y_pred