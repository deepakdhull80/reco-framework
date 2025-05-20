
python_path="/Users/deepak.dhull/mini/envs/reco/bin/python"
#prepare data for bert4rec
$python_path data/movielen_seq_data.py "/Users/deepak.dhull/data/recsys/ml-1m"

# Train bert4rec model
export HYDRA_FULL_ERROR=1
$python_path main.py -cn train-bert4rec