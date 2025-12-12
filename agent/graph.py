from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan": response}


graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.set_entry_point('planner')

agent = graph.compile()

user_prompt = "Create a simple calculator web application"

result = agent.invoke({"user_prompt": user_prompt})
print(result)
