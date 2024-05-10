from enum import Enum


class ModelType(Enum):
    RETRIEVAL = 'retrieval'
    RANKER = 'ranker'

class Environment(Enum):
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'

class TrainingStrategyNames(Enum):
    ONE_GPU = 'one_gpu'
    ALL_REDUCE = 'all_reduce'
    ACCELERATE = 'accelerate'