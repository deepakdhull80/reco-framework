from enum import Enum


class ModelType(Enum):
    RETRIEVAL = 'retrieval'
    RANKER = 'ranker'

class Environment(Enum):
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'