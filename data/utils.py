import re
import os
import string
from typing import Tuple, Dict
import numpy as np
import pandas as pd

def train_test_split(df: pd.DataFrame, sort_by_key: str, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(by=sort_by_key)
    _n = df.shape[0]
    n_train = int(_n * train_size)
    
    train, test = df.iloc[: n_train], df.iloc[n_train:]
    return train, test

def dat_to_df(path: str, cols: list, col_dtypes: dict, format: str='r') -> pd.DataFrame:
    with open(path, format) as f:
        content = f.readlines()

    def fn(data):
        if format == 'rb':
            data = data.decode('latin-1').encode('utf-8').decode('utf-8')
        data = data.replace("\n", "")
        return data.split("::")

    content = list(map(lambda x: fn(x), content))
    df = pd.DataFrame(content, columns=cols)
    for col in cols:
        df[col] = df[col].astype(col_dtypes[col])
    return df

def get_year(s):
    _s = re.findall(r'\d{4}', s)
    return int(_s[-1])

class GloVeWrapper:
    def __init__(self, base_path: str):
        self.embedding_index = self.get_GloVe_mapping(base_path=base_path)

    def get_GloVe_mapping(self, base_path: str) -> Dict[str, np.ndarray]:
        if not os.path.exists("base_path"):
            os.system(f"wget https://nlp.stanford.edu/data/glove.6B.zip -P {base_path}/")
            os.system(f"unzip {base_path}/glove.6B.zip")

        glove_file_path = f"{base_path}/glove.6B.100d.txt"
        embedding_index = {}
        with open(glove_file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs
        
        return embedding_index

    def get_embedding(self, text, emb_size = 100):
        # Tokenize the movie title
        tokens = text.split()
        
        # Initialize an empty list to store word vectors
        word_vectors = []
        
        # Iterate over tokens and get word vectors
        for token in tokens:
            token = token.lower().strip()
            token = "".join([t for t in token if t not in string.punctuation])
            # Check if token exists in the vocabulary
            if token in self.embedding_index:
                word_vectors.append(self.embedding_index[token])
        
        # Calculate the average of word vectors
        if len(word_vectors):
            # Convert list of vectors to numpy array
            word_vectors = np.array(word_vectors)
            # Average the word vectors along axis 0 (rows)
            text_embedding = np.mean(word_vectors, axis=0)
            text_embedding = text_embedding/ np.linalg.norm(text_embedding)
            return text_embedding[:emb_size]
        else:
            return np.zeros(emb_size, dtype='float32')