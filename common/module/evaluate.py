import torch
from tqdm import tqdm

def evaluate(model, dataloader, mode='Eval', eval_k=10):
    """
    Evaluate the model using Hit Rate@K (HR@K) and Normalized Discounted Cumulative Gain@K (NDCG@K).
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader for evaluation data.
        mode (str): Mode of evaluation, e.g., 'Eval' or 'Test'.
        eval_k (int): The number of top predictions to consider for HR@K and NDCG@K.
    
    Returns:
        Tuple[float, float]: Average HR@K and NDCG@K scores.
    """
    model.eval()
    total_hr_k = 0.0
    total_ndcg_k = 0.0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"{mode}uating", leave=False)
        for batch in progress_bar:
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            labels = batch['history_feature'].to(model.device)
            logits, mask_positions = model(batch, train=False)
            device = logits.device
            
            # Find positions where we need to make predictions (masked positions)
            # mask_positions = (input_ids == mask_token) & (labels != pad_token)
            
            if not torch.any(mask_positions):
                continue
                
            # Get logits and true labels at mask positions
            target_labels = labels[:, -1].reshape(-1)
            # target_labels = labels[mask_positions]
            masked_logits = logits.clone()
            
            if target_labels.numel() > 0:
                # Get top-k predictions
                _, top_k_indices = torch.topk(masked_logits, k=eval_k, dim=-1)
                
                # Calculate Hit Rate@K
                target_labels_expanded = target_labels.unsqueeze(1)
                hits = (top_k_indices == target_labels_expanded).any(dim=1)
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
                    f'HR@{eval_k}': f"{avg_hr:.4f}",
                    f'NDCG@{eval_k}': f"{avg_ndcg:.4f}"
                })
                
    # Calculate final metrics
    avg_hr_k = total_hr_k / total_samples if total_samples > 0 else 0.0
    avg_ndcg_k = total_ndcg_k / total_samples if total_samples > 0 else 0.0
    
    return avg_hr_k, avg_ndcg_k