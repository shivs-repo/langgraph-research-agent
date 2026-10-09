import os
import json
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from langsmith import traceable
from graph.state import ResearchState

load_dotenv()

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

REASONING_SYSTEM = """You are a research analyst. Given a research topic and previously gathered data, you must:
1. Evaluate if the existing data sufficiently answers the research topic
2. Generate refined search queries if more research is needed

Respond ONLY in this JSON format:
{
  "is_sufficient": true/false,
  "reasoning": "your evaluation of the current data",
  "search_queries": ["query1", "query2", "query3"]
}

If is_sufficient is true, search_queries can be empty.
Focus on finding academic research papers, studies, and scholarly articles."""


@traceable(name="Reasoning Agent", project_name="LangGraphMiocrodeg")
def reasoning_agent(state: ResearchState) -> ResearchState:
    iteration = state.iteration + 1
    existing_results = state.search_results

    results_summary = ""
    if existing_results:
        parts = ["\n\nPreviously gathered data:\n"]
        for r in existing_results[-10:]:
            parts.append(f"- Title: {r.get('title', 'N/A')}\n  URL: {r.get('url', '')}\n  Content: {r.get('content', '')[:300]}\n\n")
        results_summary = "".join(parts)

    user_msg = f"Research Topic: {state.topic}\nIteration: {iteration}/{state.max_iterations}{results_summary}"

    response = openrouter.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
    )

    parsed = json.loads(response.choices[0].message.content)
    is_sufficient = parsed.get("is_sufficient", False)
    reasoning = parsed.get("reasoning", "")
    search_queries = parsed.get("search_queries", [])

    node_output = {
        "iteration": iteration,
        "node": "Reasoning Agent",
        "is_sufficient": is_sufficient,
        "reasoning": reasoning,
        "search_queries": search_queries,
    }

    return state.model_copy(update={
        "iteration": iteration,
        "search_queries": search_queries,
        "reasoning_output": reasoning,
        "is_sufficient": is_sufficient,
        "node_outputs": state.node_outputs + [node_output],
    })


@traceable(name="Web Search Agent", project_name="LangGraphMiocrodeg")
def web_search_agent(state: ResearchState) -> ResearchState:
    queries = state.search_queries
    all_results = list(state.search_results)
    search_summaries = []

    for query in queries[:3]:  # limit to 3 queries per iteration
        try:
            response = tavily.search(
                query=f"{query} research paper",
                search_depth="advanced",
                max_results=3,
                include_domains=["arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "semanticscholar.org", "researchgate.net"],
            )
            results = response.get("results", [])
            all_results.extend(results)
            search_summaries.append({"query": query, "results": results})
        except Exception as e:
            search_summaries.append({"query": query, "error": str(e), "results": []})

    node_output = {
        "iteration": state.iteration,
        "node": "Web Search Agent",
        "searches": search_summaries,
    }

    return state.model_copy(update={
        "search_results": all_results,
        "node_outputs": state.node_outputs + [node_output],
    })


@traceable(name="Report Generator", project_name="LangGraphMiocrodeg")
def generate_report(state: ResearchState) -> ResearchState:
    results_text = "\n\n".join(
        f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:500]}"
        for r in state.search_results
    )

    response = openrouter.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a research paper analyst. Synthesize the gathered research data into a comprehensive, well-structured research summary with key findings, methodologies, and references.",
            },
            {
                "role": "user",
                "content": f"Topic: {state.topic}\n\nGathered Research Data:\n{results_text}\n\nGenerate a comprehensive research summary.",
            },
        ],
        max_tokens=2048,
    )

    report = response.choices[0].message.content

    node_output = {
        "iteration": state.iteration,
        "node": "Report Generator",
        "report": report,
    }

    return state.model_copy(update={
        "final_report": report,
        "node_outputs": state.node_outputs + [node_output],
    })


def should_continue(state: ResearchState) -> str:
    if state.is_sufficient:
        return "generate_report"
    if state.iteration >= state.max_iterations:
        return "generate_report"
    return "web_search"
