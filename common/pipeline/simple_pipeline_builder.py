from typing import List
from common.pipeline_builder import TrainerPipeline
from datetime import datetime

class SimpleTrainerPipeline(TrainerPipeline):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def execute(self):
        # Get DataLoader
        train_dl, val_dl = self.data_loader_strategy.get_dataloader()
        
        # init model
        model = self.model_builder.build(device=self.device)
        model.to(self.device)
        # Start Training
        self.train(train_dl, val_dl, model)
        return

    def train(
            self,
            train_dl,
            val_dl,
            model
    ) :
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
        Exports the model state_dict, evaluation results, and inference results to files.
        
        Args:
            state_dict: The state dictionary of the model.
            eval_result: The evaluation results.
            inference_result: The inference results.
            training_done (bool): Flag indicating if training is complete.
        """
        export_dir = "exported_models"
        os.makedirs(export_dir, exist_ok=True)

        # Generate a timestamp to avoid overwriting files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the model state_dict
        model_path = os.path.join(export_dir, f"model_state_{timestamp}.pth")
        torch.save(state_dict, model_path)

        # Save the evaluation results
        eval_result_path = os.path.join(export_dir, f"eval_results_{timestamp}.json")
        with open(eval_result_path, "w") as f:
            json.dump(eval_result, f, indent=4)

        # Save the inference results
        inference_result_path = os.path.join(export_dir, f"inference_results_{timestamp}.json")
        with open(inference_result_path, "w") as f:
            json.dump(inference_result, f, indent=4)

        # Log export completion
        print(f"Model exported to {model_path}")
        print(f"Evaluation results exported to {eval_result_path}")
        print(f"Inference results exported to {inference_result_path}")

        if training_done:
            print("Training is complete. All artifacts have been exported.")