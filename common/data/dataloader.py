import os
from typing import List
from pydantic import BaseModel
from enum import Enum

class DataLoaderType:
    SIMPLE: str = 'simple'

class DataLoaderStrategy(BaseModel):
    name: str
    
    def get_file_paths(self, path: str) -> List:
        raise NotImplementedError()


class SimpleDataLoaderStrategy(DataLoaderStrategy):
    name: str = 'simple'
    
    def get_file_paths(self, path: str, file_format: str) -> List:
        files = [f"{path}/{p}" for p in os.listdir(path) if p.endswith(file_format)]
        assert len(files) != 0, "File not found: {}".format(path)
        
        return files