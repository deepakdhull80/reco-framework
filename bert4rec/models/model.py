from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score
    )

from common.module.layers import TransformerLayer, LinearLayer
from common.model import ModelWrapper
from common.data.feature_config import FeatureType
from bert4rec.model_config import Bert4RecConfig



class Bert4RecModule(nn.Module):
    def __init__(self, model_config: Bert4RecConfig) -> None:
        super().__init__()
        self.query_features = model_config.get_query_features()
        self.history_feature = model_config.features.get_feature_by_name(model_config.history_feature_name)
        self.items_cardinality = self.history_feature.cardinality
        self.latent_dim = model_config.latent_dim
        self.padding_key = model_config.padding_key
        self.mask_key = model_config.mask_key
        self.item_embedding_table = nn.Embedding(self.items_cardinality, self.latent_dim)
        self.tl_heads = model_config.tl_heads
        
        self.transformer_layers = nn.ModuleList([TransformerLayer(self.latent_dim, self.tl_heads, dropout=0.1, bias=True) for _ in range(model_config.tl_layers)])
        self.projection_layer = LinearLayer(self.latent_dim, [self.latent_dim*2], bias=True)
    
    def _item_logits(self, x: torch.Tensor, cloze_masking: torch.Tensor = None) -> torch.Tensor:
        # item table lookup (cardinality, dimension)
        b, s, d = x.shape
        if cloze_masking is not None:
            # during training
            x = x.view(-1, d)
            cloze_masking = cloze_masking.view(-1)
            x = x[cloze_masking]
        else:
            # expecting during ineference
            x = x[:, -1, :]
            x = x.view(b, d)
        
        x = torch.einsum("bd,cd->bc", x, self.item_embedding_table.weight)
        return x    
    
    def forward(self, batch: nn.ModuleDict, cloze_masking: torch.Tensor=None) -> torch.Tensor:
        """ return the output of bert4rec model.

        Args:
            batch (nn.ModuleDict):

        Returns:
            torch.Tensor: logits for every head
        """
        
        # history feature (B, S)
        history_feature_ids = batch[self.history_feature.name]
        device = history_feature_ids.device
        
        B, S = history_feature_ids.shape
        history_mask = (history_feature_ids == self.padding_key).to(device)
        
        if cloze_masking is not None:
            history_feature_ids = history_feature_ids.masked_fill(cloze_masking, self.mask_key)
        else:
            pass
        
        # item id representation
        x = self.item_embedding_table(history_feature_ids)
        
        # pass through transformer
        for mod in self.transformer_layers:
            x = mod(x, attn_mask=history_mask)
        
        # projection layer: (B, S, D)
        x = self.projection_layer(x)
        logits = self._item_logits(x, cloze_masking=cloze_masking)
        return logits.sigmoid(dim=-1)

class Bert4RecModel(ModelWrapper):
    def __init__(self, model_config: Bert4RecConfig, device: str) -> None:
        super().__init__(model_config, device)
        self.model_config = model_config
        self.history_feature_name = model_config.history_feature_name
        self.mask_key = model_config.mask_key
        self.padding_key = model_config.padding_key
        self.target_id_name = model_config.target_id_name
        self.bert_module = Bert4RecModule(model_config)
        self.device = device
        self.cloze_masking_factor = model_config.cloze_masking_factor
        
        # initialize model weights
        self._initialize_bert4rec_weights()
    
    def _initialize_bert4rec_weights(mean=0.0, std=0.02):
        for module in self.bert_module.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                init.normal_(module.weight, mean=mean, std=std)
                if hasattr(module, 'bias') and module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                init.constant_(module.bias, 0.0)
                init.constant_(module.weight, 1.0)
    
    def _transform_device(self, batch: nn.ModuleDict) -> nn.ModuleDict:
        # TODO: need to fix this based upon multi-gpu, device transfer with global variables
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            else:
                batch[key] = value  # Preserve non-tensor values as it is
        return batch
    
    def forward(self, batch: nn.ModuleDict, train:bool=False):
        self._transform_device(batch)
        if train:
            history_feature_ids = batch[self.history_feature_name]
            B, S = history_feature_ids.shape
            history_mask = (history_feature_ids == self.padding_key).to(device)
            cloze_masking = (torch.rand(B, S) < self.cloze_masking_factor).int().to(self.device)
            cloze_masking = cloze_masking.masked_fill(history_mask, 0)
        else:
            cloze_masking = None
        
        bert_output = self.bert_module(batch, cloze_masking=cloze_masking)
        return bert_output, cloze_masking
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        bert_output, cloze_masking = self.forward(batch)
        device = bert_output.device
        cloze_masking = cloze_masking.view(-1)
        labels = batch[self.history_feature_name].view(-1)[cloze_masking].to(device)
        loss = torch.nn.functional.cross_entropy(bert_output, labels)
        
        with torch.no_grad():
            metrics = {}
            p_out = torch.argmax(bert_output.cpu(), dim=-1)
            metrics['micro_f1_score'] = f1_score(labels.cpu().numpy(), p_out.numpy(), average='micro')
        return loss, metrics
        
    @torch.no_grad()
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        
        bert_output, cloze_masking = self.forward(batch)
        device = bert_output.device
        cloze_masking = cloze_masking.view(-1)
        labels = batch[self.history_feature_name].view(-1)[cloze_masking].to(device)
        loss = torch.nn.functional.cross_entropy(bert_output, labels)
        
        metrics = {}
        p_out = torch.argmax(bert_output.cpu(), dim=-1)
        metrics['micro_f1_score'] = f1_score(labels.cpu().numpy(), p_out.numpy(), average='micro')
        return loss, metrics
    
    def get_optimizer_clz(self, clz):
        if clz == 'Adam':
            return torch.optim.Adam
        elif clz == 'SparseAdam':
            return torch.optim.SparseAdam
        else:
            raise ValueError(f"{clz} is not available.")
        