from typing import List
from common.data.dataloader import DataLoaderStrategy
from common.model import ModelBuilder
from common.trainer.training_strategy import TrainingStrategy

class PipelineOptions:
    SIMPLE = 'simple'

class TrainerPipeline:
    """ 
    Initialize TrainerPipeline with a training_strategy and data_loader
    """

    def __init__(
            self,
            model_builder: ModelBuilder,
            training_strategy,
            data_loader_strategy: DataLoaderStrategy,
            device: str, 
            *args, **kwargs
    ):
        self.model_builder = model_builder
        self.training_strategy = training_strategy
        self.data_loader_strategy = data_loader_strategy
        self.device = device

    def _get_train_data_paths(self):
        raise NotImplementedError()

    def _get_val_data_paths(self) -> List[str]:
        raise NotImplementedError()
    
    def _get_extra_day_val_data_paths(self) -> List[str]:
        raise NotImplementedError()

    def execute(self):
        # initialise trackers
        # get paths
        # Start training
        # calculate eval metrics and save them
        raise NotImplementedError()

    def train(
            self,
            train_data_paths: List[str],
            val_data_paths: List[str]
    ) :
        raise NotImplementedError()

    def eval_model(self, model, inference_result):
        raise NotImplementedError()

    def run_inference(self, model):
        raise NotImplementedError()

    def export_model(
            self,
            export_dir,
            state_dict,
            eval_result,
            inference_result,
            training_done: bool = False,
    ):
        """
        Saves the inference model (torch-scripted) to S3.
        If result_df / score_df is not None:
        - saves result_df score_df to a csv file
        - initializes HNSW index with train item DF and saves the index to S3
        - additionally saves a Dict with a mapping from item_id to hnsw index
        """
        raise NotImplementedError()


