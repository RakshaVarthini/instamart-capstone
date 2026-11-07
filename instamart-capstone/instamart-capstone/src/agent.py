"""Part D tasks 5-8: tools, ReAct loop, safe stopping, checkpoint/resume."""
import json, os, sqlite3, time
from datetime import datetime, timezone
import mock_llm
from src.rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = "state"
AGENT_SYSTEM = ("You are a refund AGENT. Use the tools get_order, search_policy, calculate_refund, issue_refund and "
                "escalate_to_human. Finish with STATUS: explanation where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND.")
CURRENT = {"run_id": None, "scenario_id": None}     # context stamped onto ledger lines

class SimulatedCrash(Exception):
    pass

# ---------------- tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": retrieve(query, 3)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool, "order_id": order_id,
            "refund_inr": refund_inr, "run_id": CURRENT["run_id"], "scenario_id": CURRENT["scenario_id"]}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")

def issue_refund(order_id, refund_inr):
    if refund_inr > 1000:                            # hard safety guard: never auto-issue above INR 1000
        return {"error": "refund above INR 1000 must be escalated to a human"}
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery minutes in the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the top matching chunks.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# ---------------- loop ----------------
def _checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    data = {k: st[k] for k in ("run_id", "scenario_id", "user_message", "messages", "tokens_used", "steps",
                              "llm_calls", "tools_executed", "max_iterations", "token_limit", "time_limit_seconds")}
    data["step"] = len(st["steps"])
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)

def _fallback(st, status, reason):
    last = st["steps"][-1] if st["steps"] else None
    summary = (f"HANDOFF to human agent: automated run stopped ({reason}) after {len(st['steps'])} tool step(s) and "
               f"{st['llm_calls']} model call(s). Tools run: {', '.join(st['tools_executed']) or 'none'}. "
               f"Last observation: {json.dumps(last['observation'], default=str)[:150] if last else 'none'}. "
               f"Customer request: {st['user_message']}")
    return {"status": status, "final_answer": summary, "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "run_id": st["run_id"]}

def _loop(st, crash_after_tools=None):
    start = time.monotonic()
    CURRENT.update(run_id=st["run_id"], scenario_id=st["scenario_id"])
    while True:
        if time.monotonic() - start >= st["time_limit_seconds"]:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {st['time_limit_seconds']}s reached")
        if st["llm_calls"] >= st["max_iterations"]:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"{st['max_iterations']} model calls without a final answer")
        r = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += r.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:       # stop before executing the requested tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit {st['token_limit']} exceeded ({st['tokens_used']})")
        msg = r.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return {"status": final.split(":")[0].strip(), "final_answer": final, "steps": st["steps"],
                    "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "run_id": st["run_id"]}
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        _checkpoint(st)
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool execution #{crash_after_tools}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "steps": [], "llm_calls": 0, "tools_executed": [],
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st)
