from pydantic import BaseModel
from typing import Any, Optional


class QueryCreate(BaseModel):
    parent_id: str
    chat_id: str
    request: str
    response: Any  # Can be string or complex object (e.g., grounding boxes)
    type: str  # stores whether it is of user or ai 
    mode: str  # which mode is the query of  


class QueryInDB(BaseModel):
    id: str
    
    parent_id: str
    chat_id: str
    request: str
    response: Any  # Can be string or complex object
    type: str


class QueryPublic(QueryInDB):
    pass


class QueryUpdate(BaseModel):
    response: Any  # Update the response field
