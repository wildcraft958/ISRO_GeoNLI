from pydantic import BaseModel


class QueryCreate(BaseModel):
    parent_id: str
    chat_id: str
    request: str
    response: str
    type: str #stores whether it is of user or ai 
    mode: str #which mode is the query of  


class QueryInDB(BaseModel):
    id: str
    parent_id: str
    chat_id: str
    request: str
    response: str
    type: str


class QueryPublic(QueryInDB):
    pass
