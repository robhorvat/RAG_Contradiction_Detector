from pydantic import BaseModel, Field
from typing_extensions import Annotated

# Pydantic v2 string constraint using Annotated + Field(pattern=...).

VerdictPattern = Annotated[
    str,
    Field(pattern=r"^(Contradictory|Supporting|Unrelated)$")
]

class Analysis(BaseModel):
    # Verdict is constrained to the supported class labels.
    verdict: VerdictPattern
    # Require a minimal justification length.
    justification: str = Field(..., min_length=10)

class LLMResponse(BaseModel):
    paper_1_claim: str
    paper_2_claim: str
    analysis: Analysis
