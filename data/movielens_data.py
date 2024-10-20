from pyspark.sql import (
    functions as F, 
    Window
    )
from pyspark.sql import types as T

from config.als_movielens import ALSMovielensConfig
from utils.spark_helper import SparkSingleton

def prepare_als_data(cfg: ALSMovielensConfig):
    spark = SparkSingleton.get_instance(cfg=cfg)
    
    data = spark.read.parquet(f"{cfg.data_cfg.src_data_path}/")
    print(f"Number of rows: {data.count()}")
    data = data.withColumn("target", (F.col("rating") >= cfg.data_cfg.rating_threshold).cast('int')) \
            .withColumn("tmp_constant", F.lit(1))
    
    window_spec = Window.partitionBy('userId').orderBy('timestamp')
    
    data = data.withColumn("request_id", F.row_number().over(window_spec))
    user_requests_df = data.groupBy('userId').count()
    data = data.join(user_requests_df, on='userId', how='inner')
    
    # create split
    data = data.withColumn("request_ratio", (F.col("request_id"))/F.col('count')) \
            .withColumn("train", F.col('request_ratio') <= cfg.data_cfg.train_size)
    train_data = data.filter(F.col('train') == True)
    eval_data = data.filter(F.col('train') == False)
    
    cols = [feature.name for feature in cfg.data_cfg.features_cfg] + ['target']
    print(f"Feature Columns: {cols}")
    train_data = train_data.select(cols)
    eval_data = eval_data.select(cols)
    
    return train_data, eval_data