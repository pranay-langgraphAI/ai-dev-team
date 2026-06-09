# state.py
from typing import TypedDict

class DevTeamState(TypedDict):
    feature_request: str
    architecture_plan: str
    source_code: str
    review_feedback: str
    status: str
    iterations: int