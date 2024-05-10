from pydantic import BaseModel

from common.constants import TrainingStrategyNames

class TrainingStrategy(BaseModel):
    strategy: TrainingStrategyNames