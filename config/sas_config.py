from config.config import Config, ModelConfig, DataConfig, FeatureConfig, FeatureKind


class SAS4RecModelConfig(ModelConfig):
    model_path: str = "checkpoint"
    
    # model parameters
    items_cardinality: int = 1_000_000
    max_length: int = 100
    # - MHA parameters
    dim: int = 32
    n_layers: int = 2
    head: int = 1
    ff_proj: list = [32, 32]
    dropout: float = 0.3    

class Filters:
    min_seq_length: int = 20

class SAS4RecDataConfig(DataConfig):
    src_data_path: str = "/Users/deepakdhull/data/recsys/ml-25m/pq"
    pad_index: int = -100
    filters = Filters()
    train_size: float = 0.8
    def __init__(self):
        super().__init__()
        features_list = [
            {
                'name': 'movieId',
                'type': 'categorical'
            },
            {
                'name': 'recent_k_rate_event',
                'type': 'categorical_list',
                'default': -100
            },
            {
                'name': 'recent_k_rate_value',
                'type': 'numerical_list',
                'default': 0.0
            }
        ]
        self.createFeaturesConfig(features_list)
    
    def createFeaturesConfig(self, feature_list):
        for feature in feature_list:
            self.features_cfg[feature['name']] = FeatureConfig(
                name = feature['name'], 
                kind = FeatureKind.get_feature_kind(feature['type']),
                default = 0 if 'default' not in feature else feature['default']
                )

class SAS4RecConfig(Config):
    random_state: int = 777
    batch_size: int = 128
    dl_shuffle: bool = True
    dl_n_workers: int = 1
    exp_name: str = "sas4rec_movie_lens"
    data_cfg = SAS4RecDataConfig()
    model_cfg = SAS4RecModelConfig()