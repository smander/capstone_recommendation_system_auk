from pydantic import BaseModel

class User(BaseModel):
    id: int
    username: str
    profile_score: float
    similarity: float
    first_name: str
    last_name: str
    # Add other fields as necessary