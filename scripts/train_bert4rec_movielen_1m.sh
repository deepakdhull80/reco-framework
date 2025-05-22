# download the movielen dataset
mkdir dataset
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
unzip dataset/ml-1m.zip -d dataset

#prepare data for bert4rec
python data/movielen_seq_data.py "./dataset/ml-1m"

# Train bert4rec model
export HYDRA_FULL_ERROR=1
python main.py -cn train-bert4rec