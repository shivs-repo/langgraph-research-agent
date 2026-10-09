import httpx
from dotenv import load_dotenv

load_dotenv()

LANGGRAPH_API_URL = "http://127.0.0.1:2024"


def show_thread(thread_id: str):
    with httpx.Client(base_url=LANGGRAPH_API_URL) as client:

        # get thread state
        resp = client.get(f"/threads/{thread_id}/state")
        if resp.status_code != 200:
            print(f"Thread not found: {resp.status_code} - {resp.text}")
            return

        data = resp.json()
        values = data.get("values", {})

        print(f"\n{'='*60}")
        print(f"Thread ID     : {thread_id}")
        print(f"Topic         : {values.get('topic', 'N/A')}")
        print(f"Iterations    : {values.get('iteration', 0)} / {values.get('max_iterations', 0)}")
        print(f"Is Sufficient : {values.get('is_sufficient', False)}")
        print(f"Next Node     : {data.get('next', [])}")
        print(f"{'='*60}")

        node_outputs = values.get("node_outputs", [])
        if node_outputs:
            print(f"\n--- Node Outputs ({len(node_outputs)} total) ---")
            for out in node_outputs:
                print(f"\n  [{out['node']}] - Iteration {out['iteration']}")
                if out["node"] == "Reasoning Agent":
                    print(f"  Sufficient : {out['is_sufficient']}")
                    print(f"  Reasoning  : {out['reasoning'][:200]}...")
                    print(f"  Queries    : {out['search_queries']}")
                elif out["node"] == "Web Search Agent":
                    for s in out.get("searches", []):
                        print(f"  Query      : {s['query']}")
                        print(f"  Results    : {len(s.get('results', []))} found")
                elif out["node"] == "Report Generator":
                    print(f"  Report     : {out['report'][:300]}...")

        final_report = values.get("final_report", "")
        if final_report:
            print(f"\n--- Final Report (first 500 chars) ---")
            print(final_report[:500])

        # get checkpoint history
        print(f"\n--- Checkpoint History ---")
        history_resp = client.get(f"/threads/{thread_id}/history")
        if history_resp.status_code == 200:
            history = history_resp.json()
            for i, checkpoint in enumerate(history):
                print(f"  [{i+1}] next={checkpoint.get('next')} | step={checkpoint.get('metadata', {}).get('step', 'N/A')}")
        else:
            print(f"  Could not fetch history: {history_resp.status_code}")


def list_threads():
    with httpx.Client(base_url=LANGGRAPH_API_URL) as client:
        resp = client.post("/threads/search", json={"limit": 10})
        if resp.status_code != 200:
            print(f"Could not list threads: {resp.status_code} - {resp.text}")
            return []
        threads = resp.json()
        print(f"\n--- Recent Threads ({len(threads)}) ---")
        for t in threads:
            print(f"  ID     : {t['thread_id']}")
            print(f"  Status : {t.get('status', 'N/A')}")
            print(f"  Created: {t.get('created_at', 'N/A')}")
            print()
        return [t["thread_id"] for t in threads]


if __name__ == "__main__":
    print("1. Enter thread ID manually")
    print("2. List recent threads")
    choice = input("\nChoice (1/2): ").strip()

    if choice == "2":
        thread_ids = list_threads()
        if thread_ids:
            thread_id = input("\nEnter thread_id from above (or press Enter for first): ").strip()
            if not thread_id:
                thread_id = thread_ids[0]
            show_thread(thread_id)
    else:
        thread_id = input("Enter thread_id: ").strip()
        show_thread(thread_id)
