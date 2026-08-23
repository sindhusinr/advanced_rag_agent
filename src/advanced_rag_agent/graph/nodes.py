from langchain_core.messages import (SystemMessage,AIMessage)

from advanced_rag_agent.generation.llm import get_llm
from advanced_rag_agent.tools.rag_tool import rag_tool
from advanced_rag_agent.tools.human_assistance import human_assistance

from advanced_rag_agent.guardrails.guardrail_manager import (
    validate_input
)

tools = [rag_tool, human_assistance]

llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


def agent_node(state):

    # Get latest user message
    user_query = state["messages"][-1].content

    # Run Guardrails
    validation = validate_input(user_query)

    if not validation["allowed"]:

        return {
            "messages": [
                AIMessage(
                    content=f"""
Request blocked by security guardrails.

Reason: {validation['reason']}
"""
                )
            ]
        }

    system_message = SystemMessage(
        content="""
You are a helpful AI assistant.

Guidelines:

- Answer normal conversation directly.
- Use rag_tool for document questions.
- Use human_assistance when information is missing.

IMPORTANT:

When rag_tool is used:

- Answer only using the retrieved document content.
- Never use source0, source1, or source2.
- For citation only use the page attached to the supporting chunk.
- Include inline citations when possible.
- Always include a Sources section at the end.

Example:

ChunkedTejas is a chunk-based approach for parallelizing trace-driven simulation [Page 6].

The technique achieved up to 5.39 speedup while maintaining only 0.2 percent error [Page 22].

Sources:
- sample.pdf (Page 6)
- sample.pdf (Page 22)

- Keep answers concise unless the user explicitly asks for a report, summary, analysis, comparison, or detailed explanation.
- If information required to complete a task is missing, use human_assistance instead of asking directly.

Examples:

weather today
→ use human_assistance

generate report
→ use human_assistance

schedule meeting
→ use human_assistance
"""
    )

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}