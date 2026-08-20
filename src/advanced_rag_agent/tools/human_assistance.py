from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def human_assistance(question: str):
    """
    Request additional information from a human.

    Use this tool whenever required information
    is missing to complete a task.

    Examples:

    weather today
    -> ask for city

    generate report
    -> ask which report

    schedule meeting
    -> ask for attendees and time
    """

    print("\n===== HUMAN ASSISTANCE CALLED =====")

    return interrupt({
        "question": question
    })