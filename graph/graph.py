from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import ResearchState, InputState
from graph.agents import reasoning_agent, web_search_agent, generate_report, should_continue

def should_continue(state: ResearchState):
    if state.is_sufficient:
        return "generate_report"
    else:
        return "web_search"

def build_graph(checkpointer=None):
    g = StateGraph(ResearchState, input=InputState)

    g.add_node("reasoning", reasoning_agent)
    g.add_node("web_search", web_search_agent)
    g.add_node("generate_report", generate_report)

    g.set_entry_point("reasoning")

    g.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "web_search": "web_search",
            "generate_report": "generate_report",
        },
    )

    g.add_edge("web_search", "reasoning")
    g.add_edge("generate_report", END)

    return g.compile(checkpointer=checkpointer)


# LangGraph Studio: no checkpointer, platform handles persistence
graph = build_graph()
