from typing import Any, Tuple, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.movielens_data import prepare_sas_data
from config import SAS4RecConfig
from config.config import FeatureKind

class MovieLens(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: SAS4RecConfig) -> None:
        self.cfg = cfg
        self.df = df
    
    def prepare_inputs(self, x) -> Dict[str, torch.Tensor]:
        inp = {}
        for f_name, feature in self.cfg.data_cfg.features_cfg.items():
            if feature.kind == FeatureKind.CATEGORICAL:
                inp[f_name] = torch.tensor(x[f_name]).unsqueeze(0)
            if feature.kind == FeatureKind.CATEGORICAL_LIST:
                inp[f_name] = F.pad(torch.tensor(x[f_name]), pad=[self.cfg.model_cfg.max_length,], value=feature.default).unsqueeze(0)
            if feature.kind == FeatureKind.NUMERICAL_LIST:
                inp[f_name] = F.pad(torch.tensor(x[f_name]), pad=[self.cfg.model_cfg.max_length,], value=feature.default).unsqueeze(0)

        return inp
        
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        return self.prepare_inputs(row)


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.concat([item[key] for item in batch], dim=0)
    return collated_batch

def get_dataset(cfg: SAS4RecConfig)-> Tuple[DataLoader, DataLoader]:
    data = prepare_sas_data(cfg)
    train, val = train_test_split(data, train_size=cfg.data_cfg.train_size, random_state=cfg.random_state)
    
    train_ds = MovieLens(train, cfg)
    val_ds = MovieLens(val, cfg)
    
    train_dl = DataLoader(train_ds, 
                        batch_size=cfg.batch_size, 
                        shuffle=cfg.dl_shuffle, 
                        num_workers=cfg.dl_n_workers, 
                        collate_fn=custom_collate_fn
                        )
    
    val_dl = DataLoader(val_ds, 
                        batch_size=cfg.batch_size, 
                        shuffle=False, 
                        num_workers=cfg.dl_n_workers, 
                        collate_fn=custom_collate_fn
                        )
    
    return train_dl, val_dl