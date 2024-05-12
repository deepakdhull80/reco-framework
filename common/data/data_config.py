from pydantic import BaseModel

class DataConfig(BaseModel):
    dataset_name: str
    description: str = ""
    base_path: str
    file_format: str