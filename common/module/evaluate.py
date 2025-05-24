import os
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
import pprint

def evaluate(model, dataloader, device, model_config, mode='Eval'):
    model.eval()
    total_hr_k = 0.0
    total_ndcg_k = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"{mode}uating", leave=False)
        for batch in progress_bar:
            input_ids = batch[model_config.history_feature_name].to(device)
            # attention_mask = batch[model_config.attention_mask].to(device)
            labels = batch[model_config.label_feature_name].to(device)
            
            logits = model(batch)
            
            # Find positions where we need to make predictions (masked positions)
            mask_positions = (input_ids == model_config.mask_key) & (labels != model_config.padding_key)
            
            if not torch.any(mask_positions):
                continue
                
            # Get logits and true labels at mask positions
            masked_logits = logits[mask_positions]
            target_labels = labels[mask_positions]
            
            if target_labels.numel() > 0:
                # Get top-k predictions
                _, top_k_indices = torch.topk(masked_logits, k=model_config.eval_k, dim=-1)
                
                # Calculate Hit Rate@K
                target_labels_expanded = target_labels.unsqueeze(1)
                hits = (top_k_indices == target_labels_expanded).sum(dim=1) > 0
                total_hr_k += hits.sum().item()
                
                # Calculate NDCG@K
                target_match = (top_k_indices == target_labels_expanded)
                ranks = torch.nonzero(target_match, as_tuple=True)
                ndcg_scores = torch.zeros(target_labels.size(0), device=device)
                
                if ranks[0].numel() > 0:
                    dcg = 1.0 / torch.log2(ranks[1].float() + 2)
                    ndcg_scores.scatter_add_(0, ranks[0], dcg)
                    
                total_ndcg_k += ndcg_scores.sum().item()
                total_samples += target_labels.size(0)
                
                # Update progress bar with current metrics
                avg_hr = total_hr_k / total_samples if total_samples > 0 else 0
                avg_ndcg = total_ndcg_k / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix({
                    f'HR@{model_config.eval_k}': f"{avg_hr:.4f}",
                    f'NDCG@{model_config.eval_k}': f"{avg_ndcg:.4f}"
                })
                
    # Calculate final metrics
    avg_hr_k = total_hr_k / total_samples if total_samples > 0 else 0.0
    avg_ndcg_k = total_ndcg_k / total_samples if total_samples > 0 else 0.0
    
    return avg_hr_k, avg_ndcg_k

def generate_recommendations(model_path, meta_path, val_df_path, dir_path, device="cuda:0", top_k=5, max_samples=50):
    """
    Generate recommendations and save the output in JSON format.

    Args:
        model_path (str): Path to the scripted model file.
        meta_path (str): Path to the mappings file (mappings.npz).
        val_df_path (str): Path to the validation DataFrame file (val.pq).
        dir_path (str): Directory path to save the JSON output.
        device (str): Device to run the model on (default: "cuda:0").
        top_k (int): Number of top predictions to consider (default: 5).
        max_samples (int): Maximum number of samples to process (default: 50).

    Returns:
        None
    """
    # Load the model
    model = torch.jit.load(model_path)
    model = model.eval().to(device)
    model.device = device

    # Load metadata and validation data
    meta = np.load(meta_path, allow_pickle=True)
    val_df = pd.read_parquet(val_df_path)

    # Prepare data
    history_feature = val_df['history_feature'].values
    history_feature = torch.from_numpy(np.array(history_feature.tolist())).to(device)

    labels = val_df['labels'].values
    labels = torch.from_numpy(np.array(labels.tolist())).to(device)

    attention_mask = val_df['attention_mask'].values
    attention_mask = torch.from_numpy(np.array(attention_mask.tolist())).to(device)

    inverse_item_map = meta['inverse_item_map'].tolist()
    original_item_to_metadata = meta['original_item_to_metadata'].tolist()

    # Mask labels
    mask = labels != 0
    history_feature[mask] = labels[mask]

    # Generate recommendations
    recommendations = []
    for i in range(history_feature.shape[0]):
        seq = history_feature[i].clone()
        label_idx = torch.argmax((history_feature[i] == 0).int()) - 1
        label = seq[label_idx].item()
        seq[label_idx] = model.mask_key

        with torch.no_grad():
            output = model({
                "history_feature": seq.to(device).view(1, -1),
                "attention_mask": attention_mask[i].to(device).view(1, -1)
            }).cpu()

        res = torch.topk(output, k=top_k)
        items = res[1][0][label_idx]
        scores = res[0][0][label_idx]
        scores = torch.softmax(scores, dim=0)

        recent_watch = [original_item_to_metadata[inverse_item_map[j.cpu().item()]] for j in history_feature[i][:label_idx][-5:]]
        factual_item = original_item_to_metadata[inverse_item_map[label]]

        recommendation = {
            "idx": i,
            "recent_watch": recent_watch,
            "factual": factual_item,
            "predicted": [
                {
                    "item_id": inverse_item_map[item.item()],
                    "metadata": original_item_to_metadata[inverse_item_map[item.item()]],
                    "score": score.item()
                }
                for score, item in zip(scores, items)
            ]
        }
        print(pprint.pformat(recommendation, compact=True).replace("'",'"'))
        print("*" * 100)
        recommendations.append(recommendation)

        if i == max_samples:
            break
    # Save recommendations to JSON
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, "recommendations.json")
    with open(output_path, "w") as f:
        json.dump(recommendations, f, default=str)

    print(f"Recommendations saved to {output_path}")


if __name__ == "__main__":
    generate_recommendations(
        model_path="artifacts/bert4rec/model_scripted_best.pt",
        meta_path="dataset/ml-1m/mappings.npz",
        val_df_path="dataset/ml-1m/val.pq",
        dir_path="artifacts/bert4rec/recommendations",
        device="cuda:0",
        top_k=5,
        max_samples=50
    )