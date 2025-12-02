from google.adk import Agent
import app.agents  # registers local llama

agent = Agent(
    name="local_helper",
    model="local/llama-8b-gguf",
    instruction="You are a concise technical assistant."
)

result = agent.run("Summarize the latest project updates.")
print(result.output)