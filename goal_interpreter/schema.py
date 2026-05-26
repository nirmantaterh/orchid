from pydantic import BaseModel, Field


class GoalInterpretation(BaseModel):
    goal: str
    subgoals: list[str] = []
    complexity: int = Field(default=5, ge=1, le=10)
    assumptions: list[str] = []
    uncertainty: int = Field(default=5, ge=0, le=10)
    initial_strategy: str = ""
    success_criteria: list[str] = []
