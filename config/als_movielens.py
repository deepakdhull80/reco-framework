from config.config import Config, ModelConfig, DataConfig, FeatureConfig, FeatureKind


class ALSMovielensModelConfig(ModelConfig):
    rank: int = 32
    max_iter: int = 30
    user_col: str = 'userId'
    item_col: str = 'movieId'
    target_col: str = 'target'
    block_size: int = 4096
    model_path: str = "checkpoint"

class ALSMovielensDataConfig(DataConfig):
    src_data_path: str = "/Users/deepakdhull/data/recsys/ml-25m/pq"
    rating_threshold: float = 3.0
    train_size: float = 0.8
    spark_cfg: dict = {
            "spark.executor.memory": "4g",
            "spark.driver.memory": "4g",
            "spark.executor.memoryOverhead": "512",
            "spark.driver.memoryOverhead": "512"
        }
    
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
            self.features_cfg[feature['name']] = FeatureConfig(name=feature['name'], kind=FeatureKind.get_feature_kind(feature['type']))

class ALSMovielensConfig(Config):
    exp_name: str = "als_movie_lens"
    data_cfg: DataConfig = ALSMovielensDataConfig()
    model_cfg: ModelConfig = ALSMovielensModelConfig()