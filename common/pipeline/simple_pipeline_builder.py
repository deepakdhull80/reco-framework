import os
import torch
import json
import numpy as np
from typing import List
from common.pipeline_builder import TrainerPipeline
from datetime import datetime

class SimpleTrainerPipeline(TrainerPipeline):
    
    def __init__(self,*args, **kwargs):
        self.artifact_dir = kwargs.get('artifact_dir', 'artifacts')
        super().__init__(*args, **kwargs)
    
    
    def execute(self):
        # Get DataLoader
        train_dl, val_dl = self.data_loader_strategy.get_dataloader()
        
        # init model
        model = self.model_builder.build(device=self.device)
        model.to(self.device)
        # Start Training
        self.train(train_dl, val_dl, model)
        self.persist_data_sample(train_dl, f"{self.artifact_dir}/train.npz")
        self.persist_data_sample(val_dl, f"{self.artifact_dir}/val.npz")
        SimpleTrainerPipeline.export_model(self.artifact_dir, model, None, None, training_done=True)
        
        return

    def train(
            self,
            train_dl,
            val_dl,
            model
    ) :
        self.training_strategy.fit(train_dl, val_dl, model)
        
    
    def persist_data_sample(self, dl, path):
        """Persist the first batch data sample from the given dataloader.

        Args:
            dl : torch dataloader
                The dataloader from which the first batch will be saved.
            path: str
                Path to save the data sample in an `.npz` file.
        """
        # Get the first batch from the dataloader
        try:
            first_batch = next(iter(dl))
        except StopIteration:
            raise ValueError("The dataloader is empty. Cannot persist data sample.")

        # Convert the batch to a dictionary of NumPy arrays
        batch_data = {}
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cpu().numpy()  # Convert tensors to NumPy arrays
            else:
                batch_data[key] = value  # Keep non-tensor data as is

        # Save the batch data to an `.npz` file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **batch_data)

        print(f"First batch data sample saved to {path}")

    def eval_model(self, model, inference_result):
        raise NotImplementedError()

    def run_inference(self, model):
        raise NotImplementedError()

    @staticmethod
    def export_model(
            export_dir: str,
            model: torch.nn.Module,
            eval_result,
            inference_result,
            training_done: bool = False,
    ):
        """
        Exports the scripted model, evaluation results, and inference results to files.
        
        Args:
            model (torch.nn.Module): The PyTorch model to be scripted and saved.
            eval_result: The evaluation results.
            inference_result: The inference results.
            training_done (bool): Flag indicating if training is complete.
        """
        # Generate a timestamp to avoid overwriting files
        if training_done:
            state = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            state = "best"

        # Save the scripted model
        model_path = os.path.join(export_dir, f"model_scripted_{state}.pt")
        
        # dummy_input = {"history_feature": torch.randint(0, 100, (1, 200))}
        # traced_model = torch.jit.trace(model, (dummy_input, torch.tensor([False]).view(1, -1), ))
        # traced_model.save(model_path)
        
        scripted_model = torch.jit.script(model)  # Script the model
        torch.jit.save(scripted_model, model_path)
        print(f"Scripted model exported to {model_path}")

        # Save the evaluation results
        if eval_result is not None:
            eval_result_path = os.path.join(export_dir, f"eval_results_{state}.json")
            with open(eval_result_path, "w") as f:
                json.dump(eval_result, f, indent=4)
            print(f"Evaluation results exported to {eval_result_path}")

        # Save the inference results
        if inference_result is not None:
            inference_result_path = os.path.join(export_dir, f"inference_results_{state}.json")
            with open(inference_result_path, "w") as f:
                json.dump(inference_result, f, indent=4)
            print(f"Inference results exported to {inference_result_path}")

        if training_done:
            print("Training is complete. All artifacts have been exported.")