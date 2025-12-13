import os
import torch
import json
import numpy as np
from typing import List
from common.pipeline_builder import TrainerPipeline
from datetime import datetime
from common.module import generate_recommendations

class SimpleTrainerPipeline(TrainerPipeline):
    
    def __init__(self, *args, **kwargs):
        self.artifact_dir = kwargs.get('artifact_dir', 'artifacts')
        self.accelerator = kwargs.get('accelerator')
        super().__init__(*args, **kwargs)
    
    
    def execute(self):
        # Get DataLoader
        train_dl, val_dl = self.data_loader_strategy.get_dataloader()
        
        # init model
        # Note: model_builder.build returns a model. 
        # With Accelerate, we don't necessarily need to move it to device manually if we prepare it later.
        # But for 'TwoTower' or others, they might need device during build? 
        # Usually build(device) implies moving it. 
        # Accelerator.prepare will move it again if needed.
        model = self.model_builder.build(device=self.device)
        
        # Start Training
        self.training_strategy.fit(train_dl, val_dl, model)
        
        # Retrieve the potentially wrapped/trained model from strategy if available
        if hasattr(self.training_strategy, 'get_trained_model'):
            model = self.training_strategy.get_trained_model()

        # Only save artifacts on main process
        # Use accelerator.is_main_process if available.
        is_main = self.accelerator.is_main_process if self.accelerator else True

        if is_main:
            self.persist_data_sample(train_dl, f"{self.artifact_dir}/train.npz")
            self.persist_data_sample(val_dl, f"{self.artifact_dir}/val.npz")
            
            # Unwrap model for export if accelerator is present
            if self.accelerator:
                unwrapped_model = self.accelerator.unwrap_model(model)
            else:
                unwrapped_model = model

            SimpleTrainerPipeline.export_model(self.artifact_dir, unwrapped_model, None, None, training_done=True)
            
            # Note: generate_recommendations might need the model on a specific device
            # unwrapped_model might be on CPU or GPU.
            # Passing device to generate_recommendations.
            
            generate_recommendations(
                model_path=f"{self.artifact_dir}/model_scripted_best.pt", # Ideally use the just exported one?
                # export_model saves 'model_scripted_{timestamp}.pt' if training_done=True
                # But here we hardcoded 'model_scripted_best.pt' in the generate call?
                # This refers to the 'best' model saved during training (in strategy).
                # Strategy saves 'model_scripted_best.pt'.
                meta_path=f"{self.data_loader_strategy.pipeline_cfg.data.base_path}/mappings.npz",
                val_df_path=f"{self.data_loader_strategy.pipeline_cfg.data.base_path}/val.pq",
                dir_path=f"{self.artifact_dir}/recommendations",
                device=self.device, # Use the device string passed to pipeline
                top_k=5,
                max_samples=50
            )
        
        # Synchronize all processes before finishing
        if self.accelerator:
            self.accelerator.wait_for_everyone()
        
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
            # If dl is prepared by Accelerate, it might be an accelerator dataloader
            # which we can iterate.
            first_batch = next(iter(dl))
        except StopIteration:
            # raise ValueError("The dataloader is empty. Cannot persist data sample.")
             print(f"Warning: The dataloader is empty. Cannot persist data sample to {path}.")
             return

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
        
        device = model.device
        # Move to CPU for scripting/saving
        model = model.to("cpu")
        scripted_model = torch.jit.script(model)  # Script the model
        torch.jit.save(scripted_model, model_path)
        print(f"Scripted model exported to {model_path}")
        
        # Restore device
        model.to(device)

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