import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client as LangSmithClient
from langgraph.checkpoint.memory import MemorySaver
from graph.graph import build_graph
from graph.state import ResearchState

st.set_page_config(page_title="Research Paper Agent", page_icon="🔬", layout="wide")
st.title("🔬 Research Paper Agentic System")
st.caption("Powered by LangGraph + Tavily + OpenRouter")

with st.sidebar:
    st.header("⚙️ Configuration")
    max_iterations = st.slider("Max Iterations", min_value=1, max_value=5, value=3)
    st.info("The agent will stop early if it finds sufficient data, or run until max iterations.")

    st.divider()
    st.markdown("**🧵 Memory Thread**")
    st.caption(f"`{st.session_state.get('thread_id', 'N/A')}`")
    if st.button("🔄 New Session", help="Clears memory and starts a fresh thread"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("**🔗 LangSmith Status**")    try:
        ls_client = LangSmithClient()
        projects = [p.name for p in ls_client.list_projects()]
        project_name = "LangGraphMiocrodeg"
        if project_name in projects:
            st.success(f"✅ Connected — project `{project_name}` found")
        else:
            st.warning(f"⚠️ Connected but project `{project_name}` not found yet. It will be created on first run.")
    except Exception as e:
        st.error(f"❌ LangSmith connection failed: {e}")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

topic = st.text_input("📚 Enter Research Topic", placeholder="e.g. Transformer architecture in NLP")

if st.button("🚀 Start Research", type="primary", disabled=not topic.strip()):
    graph = build_graph(checkpointer=st.session_state.memory)

    initial_state = ResearchState(
        topic=topic.strip(),
        max_iterations=max_iterations,
    )

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.divider()
    st.subheader("📡 Agent Execution")
    st.caption(f"🧵 Thread ID: `{st.session_state.thread_id}`")

    output_placeholder = st.empty()
    iteration_containers = {}

    with st.spinner("Running research agents..."):
        final_state = None
        for step in graph.stream(initial_state, config=config, stream_mode="values"):
            final_state = step
            node_outputs = step.node_outputs if isinstance(step, ResearchState) else step.get("node_outputs", [])

            with output_placeholder.container():
                # group outputs by iteration
                iterations_map = {}
                for out in node_outputs:
                    it = out["iteration"]
                    if it not in iterations_map:
                        iterations_map[it] = []
                    iterations_map[it].append(out)

                for it_num, outputs in iterations_map.items():
                    with st.expander(f"🔄 Iteration {it_num}", expanded=(it_num == max(iterations_map.keys()))):
                        cols = st.columns(len(outputs))
                        for col, out in zip(cols, outputs):
                            node_name = out["node"]
                            with col:
                                st.markdown(f"**🤖 {node_name}**")

                                if node_name == "Reasoning Agent":
                                    status = "✅ Sufficient" if out["is_sufficient"] else "🔍 Need More Data"
                                    st.markdown(f"**Status:** {status}")
                                    st.markdown(f"**Evaluation:**\n{out['reasoning']}")
                                    if out["search_queries"]:
                                        st.markdown("**Search Queries:**")
                                        for q in out["search_queries"]:
                                            st.markdown(f"- `{q}`")

                                elif node_name == "Web Search Agent":
                                    for search in out.get("searches", []):
                                        st.markdown(f"**Query:** `{search['query']}`")
                                        if "error" in search:
                                            st.error(f"Error: {search['error']}")
                                        else:
                                            for r in search.get("results", []):
                                                st.markdown(f"📄 [{r.get('title', 'No title')}]({r.get('url', '')})")
                                                st.caption(r.get("content", "")[:200] + "...")
                                        st.divider()

                                elif node_name == "Report Generator":
                                    st.markdown(out.get("report", ""))

    # Final Report
    if final_state:
        report = final_state.final_report if isinstance(final_state, ResearchState) else final_state.get("final_report", "")
        is_sufficient = final_state.is_sufficient if isinstance(final_state, ResearchState) else final_state.get("is_sufficient", False)
        iteration = final_state.iteration if isinstance(final_state, ResearchState) else final_state.get("iteration", 0)

        if report:
            st.divider()
            st.subheader("📋 Final Research Report")

            exit_reason = "✅ Data was sufficient" if is_sufficient else f"🔁 Reached max iterations ({max_iterations})"
            st.info(f"**Exit Reason:** {exit_reason} | **Total Iterations:** {iteration}")

            st.markdown(report)

            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name=f"research_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown",
            )
