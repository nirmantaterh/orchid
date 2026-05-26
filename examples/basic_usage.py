import os
from goal_interpreter import GoalInterpreter
from goal_interpreter.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
)
gi = GoalInterpreter(provider)
result = gi.interpret("Build me a todo app with user authentication")
print(result.model_dump_json(indent=2))
