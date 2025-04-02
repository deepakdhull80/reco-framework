from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score
    )

from common.model import ModelWrapper
from common.data.feature_config import FeatureType
from bert4rec.model_config import Bert4RecConfig

class ItemTower(nn.Module):
    def __init__(self, model_config: Bert4RecConfig) -> None:
        super().__init__()
        self.item_features = model_config.get_item_features()
        self.item_id_name = model_config.item_id_name
        i = list(filter(lambda x: (x.f_type ==  FeatureType.CATEGORICAL) and (x.name == self.item_id_name), self.item_features))[0]
        self.table_dim = 32
        self.item_cardinality = i.cardinality
        
        self.n_features = 1
        self.item_embedding_table = nn.Embedding(
            num_embeddings=self.item_cardinality, 
            embedding_dim=self.table_dim, 
            sparse=i.sparse
            )
        
        self.item_categorical_features = list(filter(lambda x: (x.f_type ==  FeatureType.CATEGORICAL) and (x.name != self.item_id_name), self.item_features))
        self.n_features += len(self.item_categorical_features)
        self.categorical_layers_dict = nn.ModuleDict({
            feature.name: nn.Embedding(feature.cardinality, self.table_dim) for feature in self.item_categorical_features
        })
        
        # item numerical features
        self.numerical_features = list(filter(lambda x: (x.f_type ==  FeatureType.NUMERICAL and not x.is_datetime ), self.item_features))
        self.numerical_feature_representations = 3
        
        self.numerical_layer = nn.Sequential(
            nn.Linear(len(self.numerical_features), self.numerical_feature_representations * self.table_dim),
            nn.ReLU(),
            nn.Linear(self.numerical_feature_representations * self.table_dim, self.numerical_feature_representations * self.table_dim)
        ) if len(self.numerical_features) >= 0 else nn.Identity()
        self.n_features += self.numerical_feature_representations if len(self.numerical_features) else 0
        
        
        # vectore features
        self.vector_features = list(filter(lambda x: x.f_type ==  FeatureType.VECTOR, self.item_features))
        self.n_features += len(self.vector_features)
        self.vector_mlp = nn.ModuleDict({
            feature.name: nn.Linear(feature.dim, self.table_dim) for feature in self.vector_features
        })
        
        # final merging layer
        self.item_embedding_dim = 32
        self.merge_layer = nn.Sequential(
            nn.Linear(self.n_features * self.table_dim, self.item_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.item_embedding_dim * 4, self.item_embedding_dim)
        )
        
        
    def forward(self, batch):
        # process high cardinality feature
        output = {}
        x = self.item_embedding_table(batch[self.item_id_name])
        output[self.item_id_name] = x
        
        # process categorical features
        for f_name, fn in self.categorical_layers_dict.items():
            output[f_name] = fn(batch[f_name])
        
        # process numerical features
        ## expecting normalize feature
        numerical_v = []
        for feature in self.numerical_features:
            numerical_v.append(batch[feature.name])
        if len(numerical_v) != 0:
            numerical_v = torch.concat(numerical_v, dim=1)
            for i, out in enumerate(self.numerical_layer(numerical_v).view(-1, self.numerical_feature_representations, self.table_dim).unbind(dim=1)):
                output[f'item_numerical_output_{i}'] = out
        
        # process vector features
        for f_name, layer in self.vector_mlp.items():
            output[f_name] = layer(batch[f_name])
        
        # merge layer
        all_feat = []
        for key in sorted(output.keys()):
            all_feat.append(output[key])
        
        all_feat = torch.concat(all_feat, dim=1)
        item_embedding = self.merge_layer(all_feat)
        return item_embedding

class QueryTower(nn.Module):
    def __init__(self, model_config: Bert4RecConfig) -> None:
        super().__init__()
        self.query_features = model_config.get_query_features()
        self.query_id_name = model_config.query_id_name
        q = list(filter(lambda x: (x.f_type ==  FeatureType.CATEGORICAL) and (x.name == self.query_id_name), self.query_features))[0]
        self.table_dim = 32
        self.query_cardinality = q.cardinality
        
        self.n_features = 1
        self.query_embedding_table = nn.Embedding(
            num_embeddings=self.query_cardinality, 
            embedding_dim=self.table_dim, 
            sparse=q.sparse
            )
        
        self.query_categorical_features = list(filter(lambda x: (x.f_type ==  FeatureType.CATEGORICAL) and (x.name != self.query_id_name), self.query_features))
        self.n_features += len(self.query_categorical_features)
        
        self.categorical_layers_dict = nn.ModuleDict({
            feature.name: nn.Embedding(feature.cardinality, self.table_dim) for feature in self.query_categorical_features
        })
        
        # query numerical features
        self.numerical_features = list(filter(lambda x: (x.f_type ==  FeatureType.NUMERICAL and not x.is_datetime ), self.query_features))
        self.numerical_feature_representations = 3
        
        self.numerical_layer = nn.Sequential(
            nn.Linear(len(self.numerical_features), self.numerical_feature_representations * self.table_dim),
            nn.ReLU(),
            nn.Linear(self.numerical_feature_representations * self.table_dim, self.numerical_feature_representations * self.table_dim)
        ) if len(self.numerical_features) >= 0 else nn.Identity()
        self.n_features += self.numerical_feature_representations if len(self.numerical_features) else 0
        
        # vectore features
        self.vector_features = list(filter(lambda x: x.f_type ==  FeatureType.VECTOR, self.query_features))
        self.n_features += len(self.vector_features)
        self.vector_mlp = nn.ModuleDict({
            feature.name: nn.Linear(feature.dim, self.table_dim) for feature in self.vector_features
        })
        
        # final merging layer
        self.query_embedding_dim = 32
        self.merge_layer = nn.Sequential(
            nn.Linear(self.n_features * self.table_dim, self.query_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.query_embedding_dim * 4, self.query_embedding_dim)
        )
        
        
    def forward(self, batch):
        # process high cardinality feature
        output = {}
        x = self.query_embedding_table(batch[self.query_id_name])
        output[self.query_id_name] = x
        
        # process categorical features
        for f_name, fn in self.categorical_layers_dict.items():
            output[f_name] = fn(batch[f_name])
        
        # process numerical features
        ## expecting normalize feature
        numerical_v = []
        for feature in self.numerical_features:
            numerical_v.append(batch[feature.name])
        if len(numerical_v) != 0:
            numerical_v = torch.concat(numerical_v, dim=1)
            for i, out in enumerate(self.numerical_layer(numerical_v).view(-1, self.numerical_feature_representations, self.table_dim).unbind(dim=1)):
                output[f'query_numerical_output_{i}'] = out
        
        # process vector features
        for f_name, layer in self.vector_mlp.items():
            output[f_name] = layer(batch[f_name])
        
        # merge layer
        all_feat = []
        for key in sorted(output.keys()):
            all_feat.append(output[key])
        
        all_feat = torch.concat(all_feat, dim=1)
        query_embedding = self.merge_layer(all_feat)
        return query_embedding

class Bert4RecModel(ModelWrapper):
    def __init__(self, model_config: Bert4RecConfig, device: str) -> None:
        super().__init__(model_config, device)
        self.model_config = model_config
        self.target_id_name = model_config.target_id_name
        self.item_tower = ItemTower(model_config)
        self.query_tower = QueryTower(model_config)
        self.label_threshold = 3
        self.device = device
    
    def _transform_device(self, batch: nn.ModuleDict) -> nn.ModuleDict:
        # TODO: need to fix this based upon multi-gpu, device transfer with global variables
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            else:
                batch[key] = value  # Preserve non-tensor values as it is
        return batch
    
    def forward(self, batch: nn.ModuleDict):
        self._transform_device(batch)
        query_output = self.query_tower(batch)
        item_output = self.item_tower(batch)
        return query_output, item_output
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        query_emb, item_emb = self.forward(batch)
        labels = (batch[self.target_id_name] > self.label_threshold).float()
        similarity_matrix = torch.einsum('ab,cd->ac', query_emb, item_emb)
        p_out = similarity_matrix.diag().sigmoid()
        loss = torch.nn.functional.binary_cross_entropy(p_out, labels)
        
        with torch.no_grad():
            metrics = {}
            metrics['train_auc'] = roc_auc_score(labels.cpu(), p_out.cpu())
            metrics['train_ap'] = average_precision_score(labels.cpu(), p_out.cpu())
        return loss, metrics
        
    @torch.no_grad()
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        query_emb, item_emb = self.forward(batch)
        labels = (batch[self.target_id_name] > self.label_threshold).float()
        similarity_matrix = torch.einsum('ab,cd->ac', query_emb, item_emb)
        p_out = similarity_matrix.diag().sigmoid()
        loss = torch.nn.functional.binary_cross_entropy(p_out, labels)
        
        metrics = {}
        metrics['val_auc'] = roc_auc_score(labels.cpu(), p_out.cpu())
        metrics['val_ap'] = average_precision_score(labels.cpu(), p_out.cpu())
        return loss, metrics
    
    def get_optimizer_clz(self, clz):
        if clz == 'Adam':
            return torch.optim.Adam
        elif clz == 'SparseAdam':
            return torch.optim.SparseAdam
        else:
            raise ValueError(f"{clz} is not available.")
        