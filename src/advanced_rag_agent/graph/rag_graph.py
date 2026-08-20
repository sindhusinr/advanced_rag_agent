from langgraph.graph import (
    StateGraph,
    START,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from advanced_rag_agent.graph.state import GraphState
from advanced_rag_agent.graph.memory import memory

from advanced_rag_agent.graph.nodes import (
    agent_node,
    tools,
)

builder = StateGraph(GraphState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
)

builder.add_edge(
    "tools",
    "agent",
)

graph = builder.compile(
    checkpointer=memory
)