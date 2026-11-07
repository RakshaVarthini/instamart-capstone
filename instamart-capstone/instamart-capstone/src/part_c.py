"""Part C: support ticket triage with guardrails (privacy, retries, validation, budget, routing)."""
import json, random, re, time
from collections import Counter
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATS = ["billing", "delivery", "technical", "fraud"]

PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ('You CLASSIFY support tickets into billing, delivery, technical or fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice and the invoice looks wrong. -> {"category": "billing", "reason": "double charge"}
Ticket: My order is late and the rider is not answering. -> {"category": "delivery", "reason": "late rider"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: There is an unauthorized transaction, I did not place it. -> {"category": "fraud", "reason": "unauthorized payment"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

def redact(text, counter=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counter is not None:
            counter[name] += n
    return text

def pii_left(text):
    return sum(len(pat.findall(text)) for _, pat, _ in PATTERNS)

def call_with_retry(messages, ticket_id, log):
    """Max 3 retries; wait before retry n (n=0,1,2) = BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            log["errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            assert BASE_DELAY * 2 ** n * 0.9 <= wait <= BASE_DELAY * 2 ** n * 1.15
            log["lines"].append(f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s reason={e}")
            time.sleep(wait)

def validate(content):
    """Returns (status, category). Never guesses."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("category"), str) and obj["category"] in CATS:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        found = [w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATS]
        if len(set(found)) == 1 and len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""

def run_pipeline(tickets, system_prompt, token_budget=None, verbose=True):
    random.seed(42)                       # reproducible jitter
    mock_llm.reset()                      # immediately before each full run
    log = {"errors": 0, "lines": []}
    prompts_sent, rows, total_tokens = [], [], 0
    for t in tickets.itertuples():
        text = t.text
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0,
               "true_category": t.true_category}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            for m in msgs:
                assert pii_left(m["content"]) == 0, "PII reached a prompt"
            prompts_sent.append(msgs)
            try:
                r = call_with_retry(msgs, t.ticket_id, log)
                used = r.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(r.choices[0].message.content)
            except TransientLLMError:
                row["status"] = "flagged_error"
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(errors=log["errors"], log=log["lines"], prompts=len(prompts_sent), total_tokens=total_tokens)
    return res

def summarise(name, res):
    ok = res[res.status.isin(["classified_json", "classified_keyword"])]
    acc = (ok.category == ok.true_category).mean()
    print(f"\n--- {name} ---")
    print("status counts:", res.status.value_counts().to_dict())
    print("model calls (tickets sent):", res.attrs["prompts"], "| TransientLLMErrors:", res.attrs["errors"],
          "| retry log lines:", len(res.attrs["log"]))
    print(f"classified {len(ok)}; accuracy {acc*100:.1f}% ({int((ok.category == ok.true_category).sum())}/{len(ok)}); "
          f"total tokens {res.tokens_used.sum()}; tokens per classified ticket {res.tokens_used.sum()/len(ok):.1f}")
    return ok

if __name__ == "__main__":
    import os; os.makedirs("outputs", exist_ok=True)
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets:", len(tickets))
    # guardrail demo
    emp = tickets.text.str.strip().eq("").sum()
    lng = tickets.text.map(est_tokens).gt(MAX_INPUT_TOKENS).sum()
    print("empty tickets:", emp, "| long tickets:", lng, "| long ticket estimate:", est_tokens(tickets[tickets.text.map(est_tokens) > 400].text.iloc[0]))
    # redaction counts
    cnt = Counter()
    red = [redact(x, cnt) for x in tickets.text]
    print("substitutions:", dict(cnt), "total", sum(cnt.values()))
    print("PII left after re-applying patterns:", sum(pii_left(x) for x in red))
    print("zero-shot prompt est tokens:", est_tokens(ZERO_SHOT), "| few-shot prompt est tokens:", est_tokens(FEW_SHOT))

    zs = run_pipeline(tickets, ZERO_SHOT)
    ok_z = summarise("zero-shot full run", zs)
    print("retry log (first 3 lines):"); [print("  ", l) for l in zs.attrs["log"][:3]]
    waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in zs.attrs["log"]]
    print(f"retry waits range {min(waits):.4f}-{max(waits):.4f}; all retry_number=0:",
          all("retry_number=0" in l for l in zs.attrs["log"]))
    print("confusion (rows=true, cols=predicted), zero-shot classified tickets:")
    print(pd.crosstab(ok_z.true_category, ok_z.category).to_string())
    fs = run_pipeline(tickets, FEW_SHOT)
    summarise("few-shot full run", fs)
    print("few-shot status counts:", fs.status.value_counts().to_dict())

    # budget demo
    bd = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    summarise("zero-shot TOKEN_BUDGET=6000", bd)
    print("budget_exhausted tickets:", int((bd.status == "budget_exhausted").sum()))
    bd_out = bd.copy()
    bd_out[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv("outputs/triage_budget_demo.csv", index=False)

    # final results with routing
    final = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    seg = pd.read_csv("outputs/customer_segments.csv")
    prem = dict(zip(seg.customer_id, seg.is_premium.astype(bool)))
    segname = dict(zip(seg.customer_id, seg.segment_name))
    def route(r):
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "flagged_error"):
            return "human_review", "NORMAL"
        if r.category == "fraud":
            return "security_team", "HIGH"
        if prem.get(r.customer_id, False):
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"
    rt = final.apply(route, axis=1, result_type="expand")
    final["route"], final["priority"] = rt[0], rt[1]
    final["segment_name"] = final.customer_id.map(segname).fillna("")
    final[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv(
        "outputs/triage_results.csv", index=False)
    print("\nfinal rows:", len(final), "| status:", final.status.value_counts().to_dict())
    print("routes:", final.route.value_counts().to_dict())
    print("priority:", final.priority.value_counts().to_dict())
    print("tickets with a customer missing from segment file:", int((final.segment_name == "").sum()))
    print("rule-validated (json/keyword):", int(final.status.isin(["classified_json", "classified_keyword"]).sum()),
          "| to humans:", int((final.route == "human_review").sum()))
