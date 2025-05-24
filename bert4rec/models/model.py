from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.final_projection = nn.Linear(self.latent_dim, self.items_cardinality, bias=model_config.bias_enable)
    
    def forward(self, batch: Dict[str, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ return the output of bert4rec model.

        Args:
            batch (Dict[str, torch.Tensor]):
            attention_mask (Optional[torch.Tensor]): 

        Returns:
            torch.Tensor: logits for every head
        """
        
        # history feature (B, S)
        history_feature_ids = batch[self.history_feature_name]
        # item id representation
        x = self.item_embedding_layer(history_feature_ids)
        x = self.transformer_layers(x, src_key_padding_mask=attention_mask)
        logits = self.final_projection(x)
        return logits

class Bert4RecModel(ModelWrapper):
    def __init__(self, model_config: Bert4RecConfig, device: str) -> None:
        super().__init__(model_config, device)
        self.model_config = model_config
        self.history_feature_name = model_config.history_feature_name
        self.attention_mask_feature_name = model_config.attention_mask_feature_name
        self.label_feature_name = model_config.label_feature_name
        self.mask_key = model_config.mask_key
        self.padding_key = model_config.padding_key
        self.target_id_name = model_config.target_id_name
        self.bert_module = Bert4RecModule(model_config)
        self.device = device
        self.cloze_masking_factor = model_config.cloze_masking_factor
        self.criterion = nn.CrossEntropyLoss()
        
        # initialize model weights
        # self._initialize_bert4rec_weights()
    
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
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._transform_device(batch)
        
        if self.attention_mask_feature_name in batch:
            attention_mask = batch[self.attention_mask_feature_name]
            attention_mask = (attention_mask == 0)
        else:
            attention_mask = None
        
        bert_output = self.bert_module(batch, attention_mask=attention_mask)
        return bert_output
    
    def train_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        metrics = {}
        labels = batch[self.label_feature_name].to(self.device)
        bert_output = self.forward(batch)
        
        mask_positions = (labels != self.padding_key)
        labels = labels[mask_positions]  # Align labels with masked positions
        logits = bert_output[mask_positions]
        
        # Calculate loss (use raw logits, not probabilities)
        loss = self.criterion(logits, labels)
        return loss, metrics
    
    @torch.no_grad()
    def val_step(self, batch: nn.ModuleDict) -> Tuple[torch.tensor, dict]:
        
        metrics = {}
        labels = batch[self.label_feature_name].to(self.device)
        bert_output = self.forward(batch)
        
        mask_positions = (labels != self.padding_key)
        labels = labels[mask_positions]  # Align labels with masked positions
        logits = bert_output[mask_positions]

        # Calculate loss (use raw logits, not probabilities)
        loss = self.criterion(logits, labels)
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
