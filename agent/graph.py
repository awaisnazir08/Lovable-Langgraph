from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_core.globals import set_verbose, set_debug

load_dotenv()

# Enable verbose logging
set_verbose(True)

# Enable full debug logging
set_debug(True)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan": response}
def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    response = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan.json()))
    if response is None:
        raise ValueError("Architect agent returned no response")
    
    response.plan = plan
    
    return {"task_plan": response}

def coder_agent(state: dict) -> dict:
    steps = state['task_plan'].implementation_steps
    current_step_idx = 0
    current_task = steps[current_step_idx]
    
    user_prompt = (
        f"Task: {current_task.task_description}\n"
    )
    system_prompt = coder_system_prompt()
    
    response = llm.invoke(system_prompt + user_prompt)
    return {"code": response.content}

graph = StateGraph(dict)

graph.add_node("planner", planner_agent)  # Add node for planner
graph.add_node("architect", architect_agent)  # Add node for architect
graph.add_node("coder", coder_agent)

graph.add_edge(start_key="planner", end_key="architect")
graph.add_edge(start_key="architect", end_key="coder")

graph.set_entry_point('planner')

agent = graph.compile()

user_prompt = "Create a simple calculator web application"

result = agent.invoke({"user_prompt": user_prompt})
print(result)

graph.add_edge("planner", "architect")