from typing import List
from pydantic import BaseModel, Field


class InputState(BaseModel):
    topic: str
    max_iterations: int = 3


class ResearchState(InputState):
    iteration: int = 0
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[dict] = Field(default_factory=list)
    reasoning_output: str = ""
    is_sufficient: bool = False
    node_outputs: List[dict] = Field(default_factory=list)
    final_report: str = ""
