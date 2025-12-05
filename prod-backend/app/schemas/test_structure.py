from pydantic import BaseModel, Field
from typing import Optional, List, Any

# --- Shared Parts ---
class ImageMetadata(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    spatial_resolution_m: Optional[float] = None

class InputImage(BaseModel):
    image_id: str
    image_url: str
    metadata: Optional[ImageMetadata] = None

# --- Query Inputs ---
class SingleQuery(BaseModel):
    instruction: str

class AttributeQueries(BaseModel):
    binary: SingleQuery
    numeric: SingleQuery
    semantic: SingleQuery

class AllQueriesInput(BaseModel):
    caption_query: SingleQuery
    grounding_query: SingleQuery
    attribute_query: AttributeQueries

# --- Query Responses ---
class ResponseWithText(BaseModel):
    instruction: str
    response: str

class ResponseWithJSON(BaseModel):
    instruction: str
    response: Any # Can be list of dicts (for bounding boxes)

class AttributeResponses(BaseModel):
    binary: ResponseWithText
    numeric: ResponseWithText # Output as text/float
    semantic: ResponseWithText

class AllQueriesResponse(BaseModel):
    caption_query: ResponseWithText
    grounding_query: ResponseWithJSON
    attribute_query: AttributeResponses

# --- Top Level ---
class TestPipelineRequest(BaseModel):
    input_image: InputImage
    queries: AllQueriesInput

class TestPipelineResponse(BaseModel):
    input_image: InputImage
    queries: AllQueriesResponse