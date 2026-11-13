"""Evaluation B: does the agent turn requests into the right search?

Each case in evals/data/agent_cases.json is a typed request with the filters
a correct agent should set ("expect") and filters it must not set ("forbid",
for example a colour filter when the shopper only said "I'd like green").
The agent runs for real (Gemini plus BigQuery); each case is repeated to
measure consistency.

    python3 -m evals.agent_eval --project YOUR_PROJECT_ID            # 30 cases x 3 runs
    python3 -m evals.agent_eval --project YOUR_PROJECT_ID --repeats 1 --only budget_k,width_feet

Checks per run:
- filters_ok: every expected filter is set to the right value, no forbidden
  filter is set. Categories: the expected category must be among those set.
  Sizes allow 3 cm, prices must match exactly, colours and materials must
  contain the expected word.
- results_ok: every returned product respects the price and size limits the
  agent set (a check on the search, independent of the agent).
- one_search: the agent searched exactly once.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

CASES_FILE = Path("evals/data/agent_cases.json")
RESULTS_FILE = Path("evals/results/agent_eval.json")
SIZE_TOLERANCE_CM = 3
FILTER_KEYS = ["categories", "min_price_inr", "max_price_inr", "colors", "materials",
               "max_width_cm", "max_depth_cm", "max_height_cm"]


def _words(values):
    return " ".join(str(v).lower() for v in (values or []))


def check_filters(expect, forbid, args):
    """Return a list of problems with the agent's tool arguments (empty means correct)."""
    problems = []
    for key, want in expect.items():
        got = args.get(key)
        if got in (None, []):
            problems.append(f"{key} not set (expected {want})")
        elif key == "categories":
            missing = [c for c in want if c not in got]
            if missing:
                problems.append(f"categories {got} lack {missing}")
        elif key in ("colors", "materials"):
            if not all(w.lower() in _words(got) for w in want):
                problems.append(f"{key} {got} lack {want}")
        elif key.endswith("_cm"):
            if abs(float(got) - float(want)) > SIZE_TOLERANCE_CM:
                problems.append(f"{key} {got} != {want}")
        elif int(got) != int(want):
            problems.append(f"{key} {got} != {want}")
    for key in forbid:
        if args.get(key) not in (None, []):
            problems.append(f"{key} set to {args[key]} but should not be")
    return problems


def _size(product):
    parts = [p.strip() for p in product.get("size_cm_w_d_h", "").split("x")]
    values = [float(p) if p not in ("", "?") else None for p in parts] + [None] * 3
    return dict(zip(("width", "depth", "height"), values[:3]))


def check_results(args, results):
    """Products that break the price or size limits the agent set."""
    broken = []
    size_keys = {"max_width_cm": "width", "max_depth_cm": "depth", "max_height_cm": "height"}
    for product in results:
        price = product.get("price_inr")
        if args.get("max_price_inr") is not None and price > args["max_price_inr"]:
            broken.append(f"{product['product_id']} price {price}")
        if args.get("min_price_inr") is not None and price < args["min_price_inr"]:
            broken.append(f"{product['product_id']} price {price}")
        size = _size(product)
        for key, axis in size_keys.items():
            limit = args.get(key)
            # Sizes are rounded to whole cm for display, so allow 1 cm.
            if limit is not None and size[axis] is not None and size[axis] > float(limit) + 1:
                broken.append(f"{product['product_id']} {axis} {size[axis]}")
    return broken


def grade_run(case, result):
    calls = [c for c in result["tool_calls"] if c["name"] == "find_similar_products"]
    if not calls:
        return {"filters_ok": False, "results_ok": None, "one_search": False,
                "problems": ["no search"], "args": None}
    first = calls[0]
    args = {k: v for k, v in (first["args"] or {}).items() if k in FILTER_KEYS}
    problems = check_filters(case.get("expect", {}), case.get("forbid", []), args)
    response = first.get("response") or {}
    broken = check_results(args, response.get("results", [])) if "results" in response else None
    return {
        "filters_ok": not problems,
        "results_ok": (not broken) if broken is not None else None,
        "one_search": len(calls) == 1,
        "problems": problems + [f"result breaks limit: {b}" for b in (broken or [])],
        "args": args,
    }


def summarize(graded):
    """graded: {case_id: [run grades]}"""
    runs = [g for grades in graded.values() for g in grades]

    def pct(values):
        values = [v for v in values if v is not None]
        return round(100 * sum(values) / len(values), 1) if values else None

    consistent = [
        len({json.dumps(g["args"], sort_keys=True) for g in grades}) == 1 for grades in graded.values()
    ]
    failures = defaultdict(list)
    for case_id, grades in graded.items():
        for g in grades:
            if g["problems"]:
                failures[case_id].append(g["problems"])
    return {
        "cases": len(graded),
        "runs": len(runs),
        "filters_correct_pct": pct(g["filters_ok"] for g in runs),
        "results_within_limits_pct": pct(g["results_ok"] for g in runs),
        "single_search_pct": pct(g["one_search"] for g in runs),
        "same_filters_every_repeat_pct": pct(consistent),
        "cases_always_correct_pct": pct(all(g["filters_ok"] for g in grades) for grades in graded.values()),
        "failures": dict(failures),
    }


async def run_cases(make_session, cases, repeats, concurrency=3):
    from agent.run import build_message

    semaphore = asyncio.Semaphore(concurrency)

    async def one(case):
        async with semaphore:
            session = make_session()
            session_id = await session.new_session()
            try:
                result = await session.send(session_id, build_message(text=case["text"]))
            except Exception as exc:  # noqa: BLE001 - count as a failed run, keep going
                result = {"reply": f"error: {exc}", "tool_calls": []}
            return case["id"], grade_run(case, result)

    outcomes = await asyncio.gather(*(one(c) for c in cases for _ in range(repeats)))
    graded = defaultdict(list)
    for case_id, grade in outcomes:
        graded[case_id].append(grade)
    return summarize(dict(graded))


def print_summary(summary):
    print(f"Cases: {summary['cases']}  Runs: {summary['runs']}")
    for key in ("filters_correct_pct", "cases_always_correct_pct", "same_filters_every_repeat_pct",
                "results_within_limits_pct", "single_search_pct"):
        print(f"  {key:32s} {summary[key]}%")
    if summary["failures"]:
        print("\nFailures:")
        for case_id, runs in summary["failures"].items():
            print(f"  {case_id}: {runs[0]}" + (f"  (in {len(runs)} runs)" if len(runs) > 1 else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", help="Comma-separated case ids")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    from agent import config
    from agent.agent import build_find_similar_agent
    from agent.cli import configure_vertex
    from agent.run import AgentSession
    from agent.search import ProductSearch

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    configure_vertex(args.project, config.MODEL_LOCATION)
    search = ProductSearch.from_environment(args.project, config.DATASET, config.BQ_LOCATION, config.MODEL_LOCATION)
    model = args.model or config.AGENT_MODEL

    summary = asyncio.run(run_cases(
        lambda: AgentSession(build_find_similar_agent(search, model=model)), cases, args.repeats))
    summary["model"] = model
    out = RESULTS_FILE.with_name(f"agent_eval_{model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Model: {model}")
    print_summary(summary)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
