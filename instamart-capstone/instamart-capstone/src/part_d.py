"""Part D runner: RAG evaluation, seven agent scenarios, stop-safely demos, crash-and-resume."""
import json, os, shutil
import pandas as pd
from src import rag, agent

# ---------------- chunking ----------------
ch = rag.load_chunks(40, 10)
print("chunks total:", len(ch), pd.Series([c["source"] for c in ch]).value_counts().sort_index().to_dict())
print("chunks with overlap 0:", len(rag.load_chunks(40, 0)))
print("top-3 for 'refund policy late delivery':", [h["source"] for h in rag.retrieve("refund policy late delivery")])

# ---------------- evaluation ----------------
ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
def evaluate(retriever, label):
    print(f"\n=== RAG evaluation: {label} ===")
    doc_ok = fact_ok = fb_ok = 0
    fails = []
    for q in ev.itertuples():
        ans, sources = rag.answer(q.question, retriever)
        if q.expected_doc == "NONE":
            fb_ok += ans.startswith("I could not find")
            if not ans.startswith("I could not find"):
                fails.append((q.question, "no fallback", ans))
        else:
            d, f = q.expected_doc in sources, q.expected_fact in ans
            doc_ok += d; fact_ok += f
            if not (d and f):
                fails.append((q.question, f"doc_in_top3={d} fact_in_answer={f}", ans))
    print(f"answerable: expected_doc in top-3 {doc_ok}/12 | answer contains expected_fact {fact_ok}/12 | unanswerable fallback {fb_ok}/2")
    for f in fails:
        print("  FAIL:", f)
evaluate(rag.Retriever(40, 10), "chunk 40 / overlap 10")
evaluate(rag.Retriever(40, 0), "chunk 40 / overlap 0")
print("\nsample answer:", rag.answer("Who must approve a refund above INR 1000?"))

# ---------------- agent scenarios ----------------
os.makedirs("outputs", exist_ok=True)
for p in (agent.LEDGER, "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl"):
    if os.path.exists(p):
        os.remove(p)
shutil.rmtree("state", ignore_errors=True); os.makedirs("state")
sc = pd.read_csv("data/raw/agent_scenarios.csv")
def refund_of(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"].get("refund_inr", 0)
    return 0
print("\n=== Seven scenarios ===")
print("| scenario | status | refund INR | expected | model calls | tokens |")
for r in sc.itertuples():
    res = agent.run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-main")
    amt = refund_of(res)
    exp_status = "FALLBACK_MAX_ITERATIONS" if r.expected_outcome == "FALLBACK" else r.expected_outcome
    assert res["status"] == exp_status, (r.scenario_id, res["status"])
    assert abs(amt - r.expected_refund_inr) < 0.01, (r.scenario_id, amt)
    print(f"| {r.scenario_id} | {res['status']} | {amt:.2f} | {r.expected_refund_inr:.2f} | {res['llm_calls']} | {res['tokens_used']} |")
    line = {"scenario_id": r.scenario_id, "status": res["status"], "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")

# ---------------- stop safely ----------------
s1 = sc.iloc[0]
def demo(label, **kw):
    res = agent.run_agent(s1.message, "S1", **kw)
    print(f"{label}: status={res['status']} llm_calls={res['llm_calls']} tokens={res['tokens_used']} steps={len(res['steps'])}")
    print("   final_answer:", res["final_answer"][:230])
    with open("outputs/agent_demos.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"demo": label, "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
print("\n=== Stop-safely demos ===")
demo("S1 token_limit=200", token_limit=200, run_id="S1-tokenlimit")
demo("S1 time_limit_seconds=0", time_limit_seconds=0, run_id="S1-timeout")

# ---------------- crash and resume ----------------
print("\n=== Crash and resume (S1) ===")
try:
    agent.run_agent(s1.message, "S1", run_id="S1-crash", crash_after_tools=2)
except agent.SimulatedCrash as e:
    print("crashed:", e)
ck = json.load(open("state/S1-crash.json"))
print("checkpoint: tools_executed =", ck["tools_executed"], "| step", ck["step"], "| tokens_used", ck["tokens_used"])
res = agent.resume_agent("S1-crash")
final_ck = json.load(open("state/S1-crash.json"))
print("resumed: status", res["status"], "| refund", refund_of(res), "| llm_calls", res["llm_calls"], "| tokens", res["tokens_used"])
print("tools executed across crash + resume, in order:", final_ck["tools_executed"])
assert final_ck["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) < 0.01
with open("outputs/agent_demos.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"demo": "S1 crash+resume", "status": res["status"], "final_answer": res["final_answer"],
                        "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
print("ledger lines:", sum(1 for _ in open(agent.LEDGER)))
