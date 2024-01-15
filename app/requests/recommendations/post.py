from pydantic import BaseModel
from typing import Union


class RecommendationParams(BaseModel):
    username: str
