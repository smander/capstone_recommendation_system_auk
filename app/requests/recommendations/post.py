from pydantic import BaseModel
from typing import Union


class RecommendationParams(BaseModel):
    user_id: str
    model: str
