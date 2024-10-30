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
    

class SAS4RecDataConfig(DataConfig):
    src_data_path: str = "/Users/deepakdhull/data/recsys/ml-25m/pq"
    
    def __init__(self):
        super().__init__()
        features_list = [
            {
                'name': 'userId',
                'type': 'categorical'
            },
            {
                'name': 'movieId',
                'type': 'categorical'
            },
            {
                'name': 'rating',
                'type': 'numerical' # create it binary 1 if rating >= 3 else 0
            }
        ]
        self.createFeaturesConfig(features_list)
    
    def createFeaturesConfig(self, feature_list):
        for feature in feature_list:
            self.features_cfg.append(
                FeatureConfig(name=feature['name'], kind=FeatureKind.get_feature_kind(feature['type']))
            )

class SAS4RecConfig(Config):
    exp_name: str = "sas4rec_movie_lens"
    data_cfg = SAS4RecDataConfig()
    model_cfg = SAS4RecModelConfig()