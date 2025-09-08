
# download dataset
mkdir dataset
if [ ! -d dataset/sharechat_recsys2023_data ]; then
    kaggle datasets download datasets/malachiugwu/sharechat-dataset -p dataset/
    unzip sharechat-dataset.zip -d dataset
fi


