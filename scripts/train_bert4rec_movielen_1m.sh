# download the movielen dataset
mkdir dataset
if [ ! -d dataset/ml-1m ]; then
    wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
    unzip dataset/ml-1m.zip -d dataset
fi

#prepare data for bert4rec
data_dir=dataset/ml-1m
if [ ! -f dataset/ml-1m/train.pq ]; then
    python data/movielen_seq_data.py $data_dir
fi

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q .; then
    device="cuda:0"
    echo "CUDA device detected: $device"

# Check for MPS availability (macOS)
elif [[ "$OSTYPE" == "darwin"* ]] && system_profiler SPDisplaysDataType | grep -q "Metal"; then
    device="mps"
    echo "MPS device detected: $device"

# Default to CPU
else
    device="cpu"
    echo "No GPU detected. Using CPU."
fi

# Train bert4rec model
export HYDRA_FULL_ERROR=1
python main.py -cn train-bert4rec data.base_path=$data_dir device=$device