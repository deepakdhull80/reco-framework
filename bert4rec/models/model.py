from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score
    )

from common.module.layers import TransformerLayer, LinearLayer
from common.model import ModelWrapper
from common.data.feature_config import FeatureType
from bert4rec.model_config import Bert4RecConfig


class BertEmbeddings(nn.Module):
    def __init__(self, num_items: int, hidden_size: int, max_len: int, pad_token: int, dropout_prob: float) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, hidden_size, padding_idx=pad_token)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        item_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = item_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Bert4RecModule(nn.Module):
    def __init__(self, model_config: Bert4RecConfig) -> None:
        super().__init__()
        self.query_features = model_config.get_query_features()
        self.history_feature = model_config.features.get_feature_by_name(model_config.history_feature_name)
        self.history_feature_name = self.history_feature.name
        self.items_cardinality = self.history_feature.cardinality
        self.max_len = self.history_feature.max_length
        self.latent_dim = model_config.latent_dim
        self.padding_key = model_config.padding_key
        self.mask_key = model_config.mask_key
        # self.item_embedding_table = nn.Embedding(self.items_cardinality, self.latent_dim)
        self.item_embedding_layer = BertEmbeddings(
            num_items=self.items_cardinality,
            hidden_size=self.latent_dim,
            max_len=self.max_len,
            pad_token=self.padding_key,
            dropout_prob=model_config.tl_dropout
        )
        self.tl_heads = model_config.tl_heads
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.latent_dim, 
            nhead=model_config.tl_heads,
            dim_feedforward=model_config.latent_dim*4, 
            dropout=model_config.tl_dropout,
            activation='gelu', 
            batch_first=True
        )
        self.transformer_layers = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=model_config.tl_layers
        )
        # self.transformer_layers = nn.ModuleList([TransformerLayer(self.latent_dim, self.tl_heads, dropout=model_config.tl_dropout, bias=model_config.bias_enable) for _ in range(model_config.tl_layers)])
        self.projection_layer = LinearLayer(self.latent_dim, [self.latent_dim*4], bias=model_config.bias_enable)
        self.final_projection = nn.Linear(self.latent_dim, self.items_cardinality, bias=model_config.bias_enable)
    
    def _item_logits(self, x: torch.Tensor, cloze_masking: Optional[torch.Tensor] = None) -> torch.Tensor:
        # item table lookup (cardinality, dimension)
        b, s, d = x.shape
        if cloze_masking is not None and cloze_masking.shape[1] == s:
            # during training
            x = x.view(-1, d)
            cloze_masking = cloze_masking.view(-1)
            x = x[cloze_masking]
        else:
            # expecting during inference
            x = x[:, -1, :]
            x = x.view(b, d)
        
        # x = torch.einsum("bd,cd->bc", x, self.item_embedding_table.weight)
        x = self.final_projection(x)
        return x    
    
    def forward(self, batch: Dict[str, torch.Tensor], cloze_masking: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ return the output of bert4rec model.

        Args:
            batch (Dict[str, torch.Tensor]):
            cloze_masking (Optional[torch.Tensor]): cloze masking for training. If None, then it is inference.

        Returns:
            torch.Tensor: logits for every head
        """
        
        # history feature (B, S)
        history_feature_ids = batch[self.history_feature_name]
        device = history_feature_ids.device
        
        B, S = history_feature_ids.shape
        history_mask = (history_feature_ids == self.padding_key).to(device)
        
        if cloze_masking is not None and cloze_masking.shape[1] == S:
            history_feature_ids = history_feature_ids.masked_fill(cloze_masking, self.mask_key)
        else:
            history_feature_ids[:, -1] = self.mask_key
        
        # item id representation
        x = self.item_embedding_layer(history_feature_ids)
        
        # pass through transformer
        # for mod in self.transformer_layers:
        #     x = mod(x, key_padding_mask=history_mask)
        x = self.transformer_layers(x, src_key_padding_mask=history_mask)
        # projection layer: (B, S, D)
        x = self.projection_layer(x)
        logits = self._item_logits(x, cloze_masking=cloze_masking)
        return logits

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
    
    def _initialize_bert4rec_weights(self, mean=0.0, std=0.02):
        for module in self.bert_module.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=mean, std=std)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def _transform_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: need to fix this based upon multi-gpu, device transfer with global variables
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            else:
                batch[key] = value  # Preserve non-tensor values as it is
        return batch
    
    def forward(self, batch: Dict[str, torch.Tensor], train: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        self._transform_device(batch)
        if train:
            history_feature_ids = batch[self.history_feature_name]
            B, S = history_feature_ids.shape
            history_mask = (history_feature_ids == self.padding_key).to(self.device)
            cloze_masking = (torch.rand(B, S) < self.cloze_masking_factor).to(torch.bool).to(self.device)
            cloze_masking = cloze_masking.masked_fill(history_mask, 0)
        else:
            cloze_masking = torch.randn(1, 1)
        
        bert_output = self.bert_module(batch, cloze_masking=cloze_masking)
        return bert_output, cloze_masking
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        labels = batch[self.history_feature_name].view(-1).to(self.device)
        bert_output, cloze_masking = self.forward(batch, train=True)
        cloze_masking = cloze_masking.view(-1)
        labels = labels[cloze_masking]  # Align labels with masked positions

        # Calculate loss (use raw logits, not probabilities)
        loss = torch.nn.functional.cross_entropy(bert_output, labels)

        with torch.no_grad():
            metrics = {}
            # Get predictions
            # p_out = torch.argmax(bert_output, dim=-1)
            # Calculate F1 score
            # metrics['micro_f1_score'] = f1_score(labels.cpu().numpy(), p_out.cpu().numpy(), average='micro')
        return loss, metrics
    
    @torch.no_grad()
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        
        labels = batch[self.history_feature_name][:, -1].to(self.device)  # Ensure this aligns with your masking strategy
        bert_output, cloze_masking = self.forward(batch)

        # Calculate loss (use raw logits, not probabilities)
        loss = torch.nn.functional.cross_entropy(bert_output, labels)

        metrics = {}
        # Get predictions
        # p_out = torch.argmax(bert_output, dim=-1)
        # Calculate F1 score
        # metrics['micro_f1_score'] = f1_score(labels.cpu().numpy(), p_out.cpu().numpy(), average='micro')
        return loss, metrics
    
    def get_optimizer_clz(self, clz):
        if clz == 'Adam':
            return torch.optim.Adam
        if clz == 'AdamW':
            return torch.optim.AdamW
        elif clz == 'SparseAdam':
            return torch.optim.SparseAdam
        else:
            raise ValueError(f"{clz} is not available.")
