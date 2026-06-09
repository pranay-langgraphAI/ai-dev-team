# nodes.py
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import DevTeamState

# Load environment variables from the local .env file
load_dotenv()

# Initialize our ultra-fast Groq model instance using the injected environment key
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# Strict Pydantic layout for the Auditor Reviewer Agent
class CodeReviewResult(BaseModel):
    passed: bool = Field(description="True if the code matches specs and contains no errors, False otherwise.")
    feedback: str = Field(description="Detailed bug reports or missing guidelines found in the script.")


def planner_node(state: DevTeamState):
    print("\n [Planner Node] Drafting technical documentation map via Groq...")
    system_msg = "You are a Principal Software Architect. Break down the user request into pseudo-code blueprints."
    user_msg = f"Feature Request: {state['feature_request']}"
    
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    return {"architecture_plan": response.content, "status": "planned", "iterations": 0}


def coder_node(state: DevTeamState):
    current_iter = state.get("iterations", 0) + 1
    print(f"\n [Coder Node] Translating specs into code... (Iteration {current_iter}) via Groq...")
    system_msg = "You are a Senior Python Developer. Output ONLY valid, raw python code block. No explanations, no markdown styling tags."
    
    user_msg = f"Architecture Plan:\n{state['architecture_plan']}"
    if state.get("status") == "review_failed":
        user_msg += f"\n\nPrevious Feedback to correct:\n{state['review_feedback']}"
        
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    return {"source_code": response.content, "iterations": current_iter, "status": "coded"}


def reviewer_node(state: DevTeamState):
    print("\n [Reviewer Node] Executing structural audits via Groq...")
    system_msg = "You are a Senior QA Specialist. Verify the source code matches requirements and looks error-free."
    user_msg = f"Plan:\n{state['architecture_plan']}\n\nGenerated Code:\n{state['source_code']}"
    
    # Enforce strict structured schema rules natively on Llama 3.3
    structured_llm = llm.with_structured_output(CodeReviewResult)
    review: CodeReviewResult = structured_llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    
    status_label = "review_passed" if review.passed else "review_failed"
    return {"review_feedback": review.feedback, "status": status_label}