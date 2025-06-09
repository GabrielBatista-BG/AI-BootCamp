from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel
from typing import List


class Topics(BaseModel):
    extracted_topics: List[str]
