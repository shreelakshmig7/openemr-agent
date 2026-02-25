"""
agent.py
--------
Healthcare AI Agent — LangChain Agent Core
-------------------------------------------
This module defines the main LangChain agent that connects Claude (Anthropic)
to the healthcare tools. The agent takes natural language queries from doctors
and uses tools to fetch real patient data and check drug interactions.

Agent Flow:
    1. Doctor asks a natural language question
    2. Claude decides which tool(s) to call
    3. Tools fetch data from mock patient database
    4. Claude synthesizes results into a coherent response

Tools Available:
    - get_patient_info: Look up patient by name
    - get_medications: Get patient's current medications
    - check_drug_interactions: Check for dangerous drug interactions

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from tools import get_patient_info, get_medications, check_drug_interactions

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "openemr-agent")

# ── Wrap tools for LangChain ──────────────────────────────────────────────────

@tool
def tool_get_patient_info(name_or_id: str) -> dict:
    """
    Look up a patient by their name OR patient ID in the healthcare system.
    Use this whenever you need a patient's full profile: name, ID, age, allergies, or conditions.
    Always call this first — even if the user provides a patient ID directly.
    Args:
        name_or_id: Full or partial patient name (e.g. "John Smith") OR a patient ID (e.g. "P001", "P002")
    Returns:
        Patient details including ID, name, age, gender, allergies, and conditions
    """
    return get_patient_info(name_or_id)


@tool
def tool_get_medications(patient_id: str) -> dict:
    """
    Get the current medication list for a patient using their patient ID.
    Always call tool_get_patient_info first to get the patient ID.
    Args:
        patient_id: The patient's ID (e.g. P001, P002, P003)
    Returns:
        List of current medications with dosage and frequency
    """
    return get_medications(patient_id)


@tool
def tool_check_drug_interactions(medication_names: list) -> dict:
    """
    Check a list of medications for dangerous drug interactions.
    Use this after getting a patient's medications to flag safety concerns.
    Args:
        medication_names: List of medication name strings to check
    Returns:
        Any dangerous interactions found with severity and recommendations
    """
    meds = [{"name": m} for m in medication_names]
    return check_drug_interactions(meds)


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a healthcare AI assistant helping doctors review 
patient medications and flag dangerous drug interactions.

Your responsibilities:
1. Look up patient information when asked
2. Retrieve current medications for patients
3. Check for dangerous drug interactions
4. Always cite the source of your information
5. Flag HIGH severity interactions with a clear WARNING
6. If you are unsure, say so clearly — never guess about medical information

CRITICAL RULE — ALWAYS CALL A TOOL FOR MEDICAL DATA:
You must NEVER answer a medical question from conversation history or memory alone.
Conversation history is ONLY used to resolve WHO the patient is (name or ID).
Every piece of medical data — allergies, medications, interactions — must come from a tool call.

This applies to follow-up questions too. Examples:
- "Does he have any allergies?" → use history to identify 'he', then call tool_get_patient_info to fetch allergies from the database
- "What medications is she on?" → use history to identify 'she', then call tool_get_medications
- "Are there any interactions?" → call tool_check_drug_interactions with fresh tool data

Always follow this order when asked about a patient:
1. First call tool_get_patient_info with the patient's name OR ID — this gives you the full profile including name, allergies, and conditions
2. Then call tool_get_medications with the patient ID from step 1 (when medications are needed)
3. Then call tool_check_drug_interactions with the medication names (when interactions are needed)
4. Synthesize all results into a clear, cited response that always includes the patient's name

If the user provides a patient ID directly (e.g. P001, P002), still call tool_get_patient_info with that ID first.
Never call tool_get_medications without first having the full patient profile from tool_get_patient_info.

Be precise, professional, and safety-focused."""


# ── Agent Factory ─────────────────────────────────────────────────────────────

def create_agent():
    """
    Create and return a LangChain agent with Claude and healthcare tools.
    Returns:
        AgentExecutor: Ready-to-use agent
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0
    )

    tools = [
        tool_get_patient_info,
        tool_get_medications,
        tool_check_drug_interactions
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )