# graph.py
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import DevTeamState
from nodes import planner_node, coder_node, reviewer_node

def route_review_verdict(state: DevTeamState) -> Literal["coder", "finish"]:
    # Safety Valve: Hard-stop infinite iteration loops to control tokens billing
    if state.get("iterations", 0) >= 3:
        print("\n [System Guardrail] Loops capped out. Force-quitting.")
        return "finish"
        
    if state.get("status") == "review_passed":
        return "finish"
        
    return "coder"

# 1. Instantiate the State Graph Blueprint
builder = StateGraph(DevTeamState)

# 2. Add our Specialized Agents
builder.add_node("planner", planner_node)
builder.add_node("coder", coder_node)
builder.add_node("reviewer", reviewer_node)

# 3. Form the Network Pathways
builder.add_edge(START, "planner")
builder.add_edge("planner", "coder")
builder.add_edge("coder", "reviewer")

# 4. Integrate the Review Feedback Loop Edge
builder.add_conditional_edges(
    "reviewer",
    route_review_verdict,
    {
        "coder": "coder",
        "finish": END
    }
)

memory_saver = MemorySaver()
compiled_squad = builder.compile(checkpointer=memory_saver)