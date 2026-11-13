import json

import pytest

from evals import agent_eval, search_eval


# --- search evaluation ------------------------------------------------------

def product(pid, category, other_ids, name=None, image=None):
    return {"product_id": pid, "category": category, "name": name or f"{category} {pid}",
            "main_image_path": image or f"m/{pid}.jpg", "other_image_ids": other_ids}


def test_pick_cases_uses_another_photo_and_caps_per_category():
    products = [product(f"S{i}", "sofa", [f"o{i}"]) for i in range(5)] + [
        product("R1", "rug", ["main-dup", "o-r1"], image="m/R1.jpg"),
        product("R2", "rug", ["unknown"]),
        product("R3", "rug", []),
    ]
    paths = {f"o{i}": f"x/o{i}.jpg" for i in range(5)} | {"main-dup": "m/R1.jpg", "o-r1": "x/r1.jpg"}
    cases = search_eval.pick_cases(products, paths, per_category=3)
    sofas = [c for c in cases if c["category"] == "sofa"]
    rugs = [c for c in cases if c["category"] == "rug"]
    assert len(sofas) == 3
    assert [(c["product_id"], c["query_image_path"]) for c in rugs] == [("R1", "x/r1.jpg")]
    assert search_eval.pick_cases(products, paths, per_category=3) == cases  # seeded


def ranked(*rows):
    return [{"product_id": p, "name": n, "category": c, "main_image_path": i} for p, n, c, i in rows]


def test_score_case_exact_variant_and_category():
    case = {"product_id": "A", "name": "Elmore Sofa, Navy", "category": "sofa", "main_image_path": "m/A.jpg"}
    rows = ranked(
        ("B", "Elmore Sofa, Grey", "sofa", "m/B.jpg"),   # variant by name stem
        ("C", "Other", "chair", "m/C.jpg"),
        ("D", "Other 2", "sofa", "m/D.jpg"),
        ("E", "Other 3", "sofa", "m/E.jpg"),
        ("A", "Elmore Sofa, Navy", "sofa", "m/A.jpg"),
    )
    scores = search_eval.score_case(case, rows)
    assert scores["exact@1"] is False and scores["exact@5"] is True
    assert scores["variant@1"] is True
    assert scores["category@1"] is True
    assert scores["exact_rank"] == 5


def test_search_summary_percentages():
    a = ({"category": "sofa"}, {"exact@1": True, "exact@5": True, "variant@1": True, "variant@5": True, "category@1": True})
    b = ({"category": "rug"}, {"exact@1": False, "exact@5": True, "variant@1": False, "variant@5": True, "category@1": False})
    summary = search_eval.summarize([a, b])
    assert summary["overall"]["exact@1"] == 50.0 and summary["overall"]["exact@5"] == 100.0
    assert summary["by_category"]["rug"]["category@1"] == 0.0


def test_search_evaluate_with_fake_search(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x/q.jpg").write_bytes(b"jpg")

    class FakeSearch:
        project, dataset = "p", "d"

        def query_vector(self, image_bytes, image_mime):
            assert image_bytes == b"jpg"
            return [1.0]

        def run_sql(self, sql, params, vector):
            assert "LIMIT 20" in sql and params == []
            return ranked(("A", "Sofa A", "sofa", "m/A.jpg"))

    cases = [
        {"product_id": "A", "name": "Sofa A", "category": "sofa", "main_image_path": "m/A.jpg", "query_image_path": "x/q.jpg"},
        {"product_id": "B", "name": "Sofa B", "category": "sofa", "main_image_path": "m/B.jpg", "query_image_path": "x/missing.jpg"},
    ]
    summary = search_eval.evaluate(FakeSearch(), cases, tmp_path, workers=1)
    assert summary["queries"] == 1 and summary["missing_query_images"] == 1
    assert summary["overall"]["exact@1"] == 100.0


# --- agent evaluation -------------------------------------------------------

@pytest.mark.parametrize(
    "expect,forbid,args,problem",
    [
        ({"max_price_inr": 25000}, [], {"max_price_inr": 25000}, None),
        ({"max_price_inr": 25000}, [], {"max_price_inr": 2500}, "max_price_inr 2500 != 25000"),
        ({"max_price_inr": 25000}, [], {}, "max_price_inr not set"),
        ({"categories": ["sofa"]}, [], {"categories": ["sofa", "accent_furniture"]}, None),
        ({"categories": ["sofa"]}, [], {"categories": ["chair"]}, "lack ['sofa']"),
        ({"max_width_cm": 122}, [], {"max_width_cm": 121.92}, None),
        ({"max_width_cm": 122}, [], {"max_width_cm": 150}, "max_width_cm 150 != 122"),
        ({"materials": ["wood"]}, [], {"materials": ["Solid Wood"]}, None),
        ({}, ["colors"], {"colors": ["green"]}, "colors set to ['green']"),
        ({}, ["colors"], {"colors": []}, None),
    ],
)
def test_check_filters(expect, forbid, args, problem):
    problems = agent_eval.check_filters(expect, forbid, args)
    if problem is None:
        assert problems == []
    else:
        assert any(problem in p for p in problems), problems


def result(pid, price, size):
    return {"product_id": pid, "price_inr": price, "size_cm_w_d_h": size}


def test_check_results_catches_limit_breaks_and_ignores_unknown_sizes():
    args = {"max_price_inr": 10000, "max_width_cm": 100, "max_height_cm": 80}
    broken = agent_eval.check_results(args, [
        result("ok", 9999, "100 x 40 x ?"),
        result("pricey", 10001, "90 x 40 x 70"),
        result("wide", 5000, "140 x 40 x 70"),
        result("tall", 5000, "90 x 40 x 82"),
    ])
    assert broken == ["pricey price 10001", "wide width 140.0", "tall height 82.0"]


def call(args, results):
    return {"name": "find_similar_products", "args": args, "response": {"results": results}}


def test_grade_run():
    case = {"expect": {"categories": ["sofa"], "max_price_inr": 25000}, "forbid": ["colors"]}
    good = agent_eval.grade_run(case, {"tool_calls": [call({"categories": ["sofa"], "max_price_inr": 25000, "description": "x"},
                                                           [result("a", 20000, "1 x 1 x 1")])]})
    assert good == {"filters_ok": True, "results_ok": True, "one_search": True, "problems": [],
                    "args": {"categories": ["sofa"], "max_price_inr": 25000}}
    twice = agent_eval.grade_run(case, {"tool_calls": [call({"categories": ["sofa"], "max_price_inr": 25000, "colors": ["grey"]}, [])] * 2})
    assert twice["filters_ok"] is False and twice["one_search"] is False
    none = agent_eval.grade_run(case, {"tool_calls": []})
    assert none["problems"] == ["no search"]


def test_grade_run_normalizes_categories_like_the_tool():
    case = {"expect": {"categories": ["rug"]}, "forbid": []}
    graded = agent_eval.grade_run(case, {"tool_calls": [call({"categories": ["Rugs"]}, [])]})
    assert graded["filters_ok"] is True and graded["args"]["categories"] == ["rug"]


def test_agent_summary_consistency():
    ok = {"filters_ok": True, "results_ok": True, "one_search": True, "problems": [], "args": {"a": 1}}
    other = dict(ok, args={"a": 2})
    bad = dict(ok, filters_ok=False, problems=["x"])
    summary = agent_eval.summarize({"c1": [ok, ok], "c2": [ok, other], "c3": [bad, bad]})
    assert summary["runs"] == 6
    assert summary["filters_correct_pct"] == 66.7
    assert summary["same_filters_every_repeat_pct"] == 66.7
    assert summary["cases_always_correct_pct"] == 66.7
    assert summary["failures"] == {"c3": [["x"], ["x"]]}


@pytest.mark.asyncio
async def test_run_cases_with_fake_sessions():
    class FakeSession:
        async def new_session(self):
            return "s"

        async def send(self, session_id, message):
            text = message.parts[0].text
            if text == "boom":
                raise RuntimeError("network")
            return {"reply": "ok", "tool_calls": [call({"categories": ["rug"]}, [])]}

    cases = [{"id": "rug", "text": "a rug", "expect": {"categories": ["rug"]}, "forbid": []},
             {"id": "err", "text": "boom", "expect": {}, "forbid": []}]
    summary = await agent_eval.run_cases(FakeSession, cases, repeats=2)
    assert summary["runs"] == 4
    assert summary["filters_correct_pct"] == 50.0
    assert summary["failures"]["err"][0] == ["no search"]


def test_agent_cases_file_is_valid():
    from agent.config import CATEGORIES

    cases = json.loads(agent_eval.CASES_FILE.read_text(encoding="utf-8"))
    assert len(cases) == 30 and len({c["id"] for c in cases}) == 30
    for case in cases:
        assert set(case["expect"]) <= set(agent_eval.FILTER_KEYS)
        assert set(case["forbid"]) <= set(agent_eval.FILTER_KEYS)
        assert set(case["expect"].get("categories", [])) <= set(CATEGORIES)
