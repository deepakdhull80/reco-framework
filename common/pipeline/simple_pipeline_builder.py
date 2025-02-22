from typing import List
from common.pipeline_builder import TrainerPipeline


class SimpleTrainerPipeline(TrainerPipeline):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def execute(self):
        train_dl, val_dl = self.data_loader_strategy.get_dataloader()
        model = self.model_builder.build(device=self.device)
        self.train(train_dl, val_dl, model)
        return

    def train(
            self,
            train_dl,
            val_dl,
            model
    ) :
        model.to(self.device)
        self.training_strategy.fit(train_dl, val_dl, model)
        
        

    def eval_model(self, model, inference_result):
        raise NotImplementedError()

    def run_inference(self, model):
        raise NotImplementedError()

    def export_model(
            self,
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