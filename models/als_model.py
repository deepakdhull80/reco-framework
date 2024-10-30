import os
import math
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from config import ALSMovielensConfig
from utils.spark_helper import SparkSingleton
from data.movielens_data import prepare_als_data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ALSTrainer:
    def __init__(self, cfg: ALSMovielensConfig) -> None:
        self.cfg = cfg
        self.spark = SparkSingleton.get_instance(cfg=cfg)
    
    def get_data(self):
        train, val = prepare_als_data(cfg=self.cfg)        
        return train, val
    
    def fit(self, train_data):
        model = ALS(
            rank=self.cfg.model_cfg.rank, 
            maxIter=self.cfg.model_cfg.max_iter,
            userCol=self.cfg.model_cfg.user_col,
            itemCol=self.cfg.model_cfg.item_col,
            implicitPrefs=True,
            seed=7,
            ratingCol=self.cfg.model_cfg.target_col,
            blockSize=self.cfg.model_cfg.block_size
            )
        os.makedirs(f"{self.cfg.model_cfg.model_path}", exist_ok=True)
        
        model = model.fit(train_data)
        
        model.save(f"{self.cfg.model_cfg.model_path}/als")
        return model
    
    def train(self):
        train_data, val_data = self.get_data()
        
        # train als model
        model = self.fit(train_data=train_data)
        
        ############ Validate Model ###############
        train_pred = model.transform(train_data)
        val_pred = model.transform(val_data)
        
        sigmoid_udf = udf(sigmoid, DoubleType())
        
        train_pred = train_pred.withColumn("prediction", sigmoid_udf('prediction'))
        val_pred = val_pred.withColumn("prediction", sigmoid_udf('prediction'))
        
        val_pred.show()
        
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="target")
        train_result = evaluator.evaluate(train_pred)
        val_result = evaluator.evaluate(val_pred)
        
        print("*"*20)
        print("TRAIN_AUROC:", train_result)
        print("VAL_AUROC:", val_result)
        print("*"*20)
        
        #TODO generate user embedding
        
        #TODO generate item embedding