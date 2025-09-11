from pydantic import BaseModel, Field
from typing_extensions import Annotated

# --- Pydantic v2 Syntax for String Constraints ---
# Instead of `constr(regex=...)`, we use `Annotated` to combine the type (str)
# with the constraint (a regular expression pattern). This is the modern,
# recommended way to use Pydantic for powerful validation.

VerdictPattern = Annotated[
    str,
    Field(pattern=r"^(Contradictory|Supporting|Unrelated)$")
]

class Analysis(BaseModel):
    # The verdict must be a string that matches our specific regex pattern.
    verdict: VerdictPattern
    # The justification must be a string with at least 10 characters.
    justification: str = Field(..., min_length=10)

class LLMResponse(BaseModel):
    paper_1_claim: str
    paper_2_claim: str
    analysis: Analysis