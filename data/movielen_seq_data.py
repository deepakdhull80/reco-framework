# --- Imports ---
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import sys

class Config:
    # --- Data Preprocessing Parameters ---
    data_path = '/data/user-files/deepak.dhull/recsys/dataset/ml-1m/ratings.dat'
    movies_path = '/data/user-files/deepak.dhull/recsys/dataset/ml-1m/movies.dat'
    # Column names after loading and renaming
    user_col = 'UserID'
    item_col = 'MovieID'
    time_col = 'Timestamp'
    # Filtering thresholds
    min_user_interactions = 5
    min_item_interactions = 5
    # Data split ratios (by user)
    train_ratio = 0.8
    eval_ratio = 0.15  # Test ratio = 1.0 - train_ratio - eval_ratio
    
    # Data parameters for model/dataset
    max_len = 200        # Maximum sequence length for model input
    mask_prob = 0.2      # Probability of masking an item for training
    mask_token = 1       # Special token ID for masking
    pad_token = 0        # Special token ID for padding
    num_items = None     # Will be set dynamically after preprocessing
    
    # Sliding window parameters for training sample generation
    min_seq_len = 5      # Minimum sequence length to consider
    stride = 5           # Stride for sliding window during training
    
    # Model parameters
    hidden_size = 256     # Embedding dimension
    num_hidden_layers = 2 # Number of Transformer layers
    num_attention_heads = 2 # Number of attention heads
    intermediate_size = hidden_size*4 # Feedforward layer size
    dropout_prob = 0.1   # Dropout probability
    
    # Training parameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 100          # Reduced from 1000 for practical training time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluation parameters
    eval_k = 10          # Evaluate HR@K and NDCG@K
    
    # Qualitative analysis parameters
    num_recommendations = 10  # Number of recommendations to show
    num_users_to_analyze = 5  # Number of users for qualitative analysis

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Modified Data Preprocessing Function ---
def preprocess_data(config):
    """
    Loads, filters, maps IDs, creates sequences, and splits data.
    Returns mappings needed for qualitative analysis.
    """
    print(f"--- Starting Data Preprocessing ---")
    print(f"Using data file: {config.data_path}")

    # 1. Load Raw Data
    if not os.path.exists(config.data_path):
        print(f"ERROR: Data file not found at '{config.data_path}'")
        print("Please ensure the file exists or update Config.data_path.")
        # Fallback to dummy data
        print("Creating minimal dummy data for demonstration purposes...")
        num_dummy_users = 100
        num_dummy_items = 50
        num_dummy_interactions = 1000
        dummy_data = {
            'user_id': [random.randint(1, num_dummy_users) for _ in range(num_dummy_interactions)],
            'item_id': [random.randint(1, num_dummy_items) for _ in range(num_dummy_interactions)],
            'timestamp': [random.randint(1, 10000) for _ in range(num_dummy_interactions)]
        }
        df = pd.DataFrame(dummy_data)
        print("Dummy data created.")
    else:
        try:
            # Load MovieLens data
            print(f"Loading data from {config.data_path} with separator '::'...")
            df = pd.read_csv(
                config.data_path,
                sep='::',
                header=None,
                engine='python',
                names=['user_id_raw', 'item_id_raw', 'rating', 'timestamp_raw'],
                dtype={'user_id_raw': np.int64, 'item_id_raw': np.int64, 'rating': np.float32, 'timestamp_raw': np.int64}
            )
            print(f"Raw data loaded: {len(df)} interactions.")

            # Select relevant columns and rename
            df = df[['user_id_raw', 'item_id_raw', 'timestamp_raw']]
            df.columns = ['user_id', 'item_id', 'timestamp']
        except Exception as e:
            print(f"Error loading or processing data from {config.data_path}: {e}")
            raise

    # 2. Load Movie Metadata (for qualitative analysis)
    original_item_to_metadata = {}
    if os.path.exists(config.movies_path):
        try:
            print(f"Loading movie metadata from {config.movies_path}...")
            movies_df = pd.read_csv(
                config.movies_path,
                sep='::',
                header=None,
                engine='python',
                names=['MovieID', 'Title', 'Genres'],
                encoding='latin-1'  # Handle special characters in movie titles
            )
            for _, row in movies_df.iterrows():
                original_item_to_metadata[row['MovieID']] = {
                    'title': row['Title'],
                    'genres': row['Genres']
                }
            print(f"Loaded metadata for {len(original_item_to_metadata)} movies.")
        except Exception as e:
            print(f"Error loading movie metadata: {e}")
            print("Qualitative analysis will show IDs instead of movie titles.")
    else:
        print(f"Movie metadata file not found at {config.movies_path}")
        print("Qualitative analysis will show IDs instead of movie titles.")

    # 3. Filter Interactions
    print(f"Filtering interactions (min_user={config.min_user_interactions}, min_item={config.min_item_interactions})...")
    last_num_interactions = len(df)
    iterations = 0
    max_iterations = 5
    while iterations < max_iterations:
        iterations += 1
        print(f"Filtering iteration {iterations}...")
        
        # Filter items first
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= config.min_item_interactions].index
        df_filtered_items = df[df['item_id'].isin(valid_items)]
        print(f"  Items: {len(df)} → {len(df_filtered_items)} interactions remaining.")
        
        # Then filter users
        user_counts = df_filtered_items['user_id'].value_counts()
        valid_users = user_counts[user_counts >= config.min_user_interactions].index
        df_filtered = df_filtered_items[df_filtered_items['user_id'].isin(valid_users)]
        print(f"  Users: {len(df_filtered_items)} → {len(df_filtered)} interactions remaining.")
        
        if len(df_filtered) == last_num_interactions:
            print("Filtering stabilized.")
            break
            
        last_num_interactions = len(df_filtered)
        df = df_filtered

    if iterations == max_iterations:
        print("Warning: Filtering did not stabilize after maximum iterations.")

    if len(df) == 0:
        raise ValueError("No interactions left after filtering. Check filter thresholds or data.")
        
    print(f"Filtered data: {len(df)} interactions for {df['user_id'].nunique()} users and {df['item_id'].nunique()} items.")

    # 4. Map IDs to Contiguous Integers
    print("Mapping User and Item IDs...")
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    # Create mapping dictionaries (both ways)
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id + 2 for new_id, old_id in enumerate(unique_items)}  # Start from 2 (0=MASK, 1=PAD)
    
    # Reverse mappings for qualitative analysis
    inverse_user_map = {v: k for k, v in user_map.items()}
    inverse_item_map = {v: k for k, v in item_map.items()}
    
    # Apply mappings
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)
    
    num_items_mapped = len(item_map) + 2  # +2 for mask and pad
    print(f"Mapped {len(user_map)} users and {len(item_map)} items (total item vocabulary size including special tokens: {num_items_mapped}).")

    # 5. Sort Interactions by User and Time
    print("Sorting interactions by user and timestamp...")
    df = df.sort_values(by=['user_id', 'timestamp'])

    # 6. Create User Sequences
    print("Creating user interaction sequences...")
    user_sequences = df.groupby('user_id')['item_id'].apply(list).to_dict()
    print(f"Created sequences for {len(user_sequences)} users.")

    # 7. Split Data (by user)
    print("Splitting users into train/validation/test sets...")
    user_ids = list(user_sequences.keys())
    random.shuffle(user_ids)
    num_users = len(user_ids)
    
    train_split_idx = int(config.train_ratio * num_users)
    eval_split_idx = int((config.train_ratio + config.eval_ratio) * num_users)
    
    train_user_ids = user_ids[:train_split_idx]
    eval_user_ids = user_ids[train_split_idx:eval_split_idx]
    test_user_ids = user_ids[eval_split_idx:]
    sequences = {uid: user_sequences[uid] for uid in user_ids}
    train_sequences = {uid: user_sequences[uid] for uid in train_user_ids}
    eval_sequences = {uid: user_sequences[uid] for uid in eval_user_ids}
    test_sequences = {uid: user_sequences[uid] for uid in test_user_ids}
    
    print(f"Data split completed:")
    print(f"  Training users: {len(train_sequences)}")
    print(f"  Validation users: {len(eval_sequences)}")
    print(f"  Test users: {len(test_sequences)}")
    
    # Store mappings for qualitative analysis
    mappings = {
        'inverse_user_map': inverse_user_map,
        'inverse_item_map': inverse_item_map,
        'original_item_to_metadata': original_item_to_metadata
    }
    
    print(f"--- Data Preprocessing Finished ---")
    return sequences, train_sequences, eval_sequences, test_sequences, num_items_mapped, mappings


# --- Modified Dataset for Using All Interactions ---
class BertRecDataset(Dataset):
    def __init__(self, user_sequences, max_len, mask_token, pad_token, mask_prob, mode='train', min_seq_len=2, stride=1):
        self.user_sequences = user_sequences
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.mask_prob = mask_prob
        self.mode = mode
        self.min_seq_len = min_seq_len
        self.stride = stride
        
        # Process user sequences into samples
        self.samples = []
        self._process_sequences()
        
    def _process_sequences(self):
        """Generate multiple training samples using sliding window approach for each user."""
        print(f"Creating {self.mode} samples with {'stride ' + str(self.stride) if self.mode == 'train' else 'last item masked'}...")
        
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) < self.min_seq_len:
                continue
                
            if self.mode == 'train':
                # For training: create multiple samples with sliding window
                for start_idx in range(0, len(sequence) - self.min_seq_len + 1, self.stride):
                    end_idx = min(start_idx + self.max_len, len(sequence))
                    if end_idx - start_idx < self.min_seq_len:
                        continue
                    self.samples.append((user_id, sequence[start_idx:end_idx]))
            else:
                # For eval/test: predict the last item
                self.samples.append((user_id, sequence))
                
        print(f"Created {len(self.samples)} {self.mode} samples from {len(self.user_sequences)} users.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        user_id, sequence = self.samples[index]
        
        if self.mode == 'train':
            # Apply random masking for training
            input_ids, labels = self._apply_train_masking(sequence)
        else:
            # For evaluation/testing, always mask the last item
            if len(sequence) < 2:
                # Handle edge case with very short sequence
                input_ids = [self.pad_token] * (self.max_len - 1) + [self.mask_token]
                labels = [self.pad_token] * self.max_len
                seq_len = 1
            else:
                # Normal case: predict the last item
                input_ids = list(sequence[:-1]) + [self.mask_token]
                labels = [self.pad_token] * (len(sequence) - 1) + [sequence[-1]]
                seq_len = len(input_ids)
                
        # Truncate if too long
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]
            seq_len = self.max_len
        else:
            seq_len = len(input_ids)
            
        # Pad if needed
        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * seq_len + [0] * padding_len
        input_ids = input_ids + [self.pad_token] * padding_len
        labels = labels + [self.pad_token] * padding_len
        
        return {
            'user_id': user_id,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        
    def _apply_train_masking(self, sequence):
        """Apply random masking to sequence for BERT-style training."""
        input_ids = list(sequence)
        labels = [self.pad_token] * len(sequence)
        
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]  # Save original ID in labels
                input_ids[i] = self.mask_token  # Mask the item
                
        return input_ids, labels

def generate_dataframe(config):
    """
    Generate a DataFrame where the `history_feature` column contains lists of the same length.

    Args:
        config (Config): Configuration object containing preprocessing parameters.

    Returns:
        pd.DataFrame: DataFrame with columns `user_id`, `history_feature`, `attention_mask`, and `labels`.
    """
    # Step 1: Preprocess data to get user sequences
    sequences, train_s, eval_s, test_s, num_items, mappings = preprocess_data(config)
    print(f"Number of items (including special tokens): {num_items}")
    # Step 2: Create a BertRecDataset instance
    dfs = []
    for seq in [train_s, eval_s, test_s]:
        dataset = BertRecDataset(
            user_sequences=seq,
            max_len=config.max_len,
            mask_token=config.mask_token,
            pad_token=config.pad_token,
            mask_prob=config.mask_prob,
            mode='train',  # Use training mode to apply random masking
            min_seq_len=config.min_seq_len,
            stride=config.stride
        )

        # Step 3: Iterate through the dataset and collect data
        data = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            data.append({
                'user_id': sample['user_id'],
                'history_feature': sample['input_ids'].tolist(),  # Convert tensor to list
                "attention_mask": sample['attention_mask'].tolist(),  # Convert tensor to list
                'labels': sample['labels'].tolist()  # Convert tensor to list
            })

        # Step 4: Create a DataFrame
        df = pd.DataFrame(data)
        dfs.append(df)
    return dfs, mappings


# Example usage
if __name__ == "__main__":
    path = sys.argv[1]
    config = Config()
    config.data_path = path + '/ratings.dat'
    config.movies_path = path + '/movies.dat'
    
    dfs, mappings = generate_dataframe(config)
    np.savez_compressed(f'{path}/mappings.npz', **mappings)
    for df, file in zip(dfs, ['train', 'val', 'test']):
        df.to_parquet(path + f'/{file}.pq')
    print(f"Generated DataFrame with {len(df)} rows.")