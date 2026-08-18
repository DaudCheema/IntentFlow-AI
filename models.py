from pydantic import BaseModel
from typing import List, Optional
from pydantic import BaseModel
from typing import List, Optional

class UserQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"


class FinalAgentResponse(BaseModel):
    answer: str
    clarifying_questions: List[str]
    suggested_questions: List[str]
    sources: List[str]