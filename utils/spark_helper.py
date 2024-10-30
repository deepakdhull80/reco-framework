from pyspark.sql import SparkSession

from config.config import Config

class SparkSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, cfg: Config):
        if cls._instance is None:
            spark = SparkSession.builder.appName(cfg.exp_name)
            for k, v in cfg.data_cfg.spark_cfg.items():
                spark = spark.config(k, v)
            
            cls._instance = spark.getOrCreate()
        return cls._instance