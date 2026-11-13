import pytest

from agent.search import Filters, ProductSearch, build_query, drop_variants, format_product


def test_query_without_filters_has_no_where():
    sql, params = build_query(Filters(), "proj", "ds", 20)
    assert "WHERE" not in sql
    assert params == []
    assert "VECTOR_SEARCH(" in sql and "'embedding'" in sql
    assert "top_k => 20" in sql and "distance_type => 'COSINE'" in sql
    assert "`proj.ds.product_embeddings`" in sql and "`proj.ds.products`" in sql


def test_all_filters_become_parameters():
    filters = Filters(
        categories=["sofa"], min_price_inr=1000, max_price_inr=20000, colors=["navy"],
        materials=["velvet"], max_width_cm=180, max_depth_cm=90, max_height_cm=85, exclude_ids=["X"],
    )
    sql, params = build_query(filters, "p", "d", 10)
    assert {name for name, _, _ in params} == {
        "categories", "min_price", "max_price", "colors", "materials",
        "max_width", "max_depth", "max_height", "exclude_ids",
    }
    assert ("max_price", "INT64", 20000) in params
    assert ("max_width", "FLOAT64", 180.0) in params
    assert "p.longest_side_cm <= @max_width" in sql
    assert "p.shortest_side_cm <= @max_depth" in sql
    assert "ARRAY_CONCAT(p.colors, p.color_families)" in sql


def test_filter_values_never_appear_in_sql():
    hostile = "navy'); DROP TABLE products; --"
    sql, params = build_query(Filters(colors=[hostile], categories=[hostile]), "p", "d", 5)
    assert hostile not in sql
    assert ("colors", "ARRAY<STRING>", [hostile]) in params


@pytest.mark.parametrize("bad", [0, 201, 5.5, "5"])
def test_top_k_is_validated(bad):
    with pytest.raises(ValueError):
        build_query(Filters(), "p", "d", bad)


def test_filters_describe_skips_unset_and_exclusions():
    assert Filters(max_price_inr=5000, exclude_ids=["x"]).describe() == {"max_price_inr": 5000}


def row(pid, name, image=None, similarity=0.9, **extra):
    base = {
        "product_id": pid, "name": name, "brand": "B", "category": "sofa", "price_inr": 19999,
        "colors": ["Navy", "Blue", "Ink", "Extra"], "materials": ["Velvet"], "styles": [],
        "longest_side_cm": 180.4, "shortest_side_cm": 86.0, "height_cm": None,
        "main_image_path": image, "similarity": similarity,
    }
    base.update(extra)
    return base


def test_drop_variants_by_image_and_name_stem():
    rows = [
        row("A", "Elmore Sofa, Navy", image="i/1.jpg"),
        row("B", "Elmore Sofa, Grey", image="i/2.jpg"),     # same stem
        row("C", "Other Sofa", image="i/1.jpg"),            # same image
        row("D", "Andover Sofa", image="i/3.jpg"),
        row("E", "Third Sofa", image="i/4.jpg"),
    ]
    assert [r["product_id"] for r in drop_variants(rows, 2)] == ["A", "D"]
    assert [r["product_id"] for r in drop_variants(rows, 10)] == ["A", "D", "E"]


def test_format_product():
    product = format_product(row("A", "Sofa", image="i/1.jpg", similarity=0.91234))
    assert product["size_cm_w_d_h"] == "180 x 86 x ?"
    assert product["colors"] == ["Navy", "Blue", "Ink"]
    assert product["similarity"] == 0.912
    assert product["image_path"] == "i/1.jpg"


class FakeRow(dict):
    pass


class FakeBigQuery:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, sql, job_config, location):
        self.calls.append((sql, job_config, location))
        rows = self.rows

        class Job:
            def result(self):
                return [FakeRow(r) for r in rows]

        return Job()


def make_search(rows):
    embedded = []

    def embed(content):
        embedded.append(content)
        return [0.6, 0.8]

    bq = FakeBigQuery(rows)
    return ProductSearch(bq, embed, "proj", "ds", "asia-south1"), bq, embedded


def test_search_embeds_query_runs_sql_and_dedupes():
    search, bq, embedded = make_search([
        row("A", "Elmore Sofa, Navy", image="i/1.jpg"),
        row("B", "Elmore Sofa, Grey", image="i/2.jpg"),
        row("C", "Andover Sofa", image="i/3.jpg"),
    ])
    results = search.search(Filters(max_price_inr=20000), image_bytes=b"jpg", image_mime="image/png",
                            text="navy sofa", top_k=5)
    assert [r["product_id"] for r in results] == ["A", "C"]
    assert len(embedded[0].parts) == 2 and embedded[0].parts[0].inline_data.mime_type == "image/png"

    sql, job_config, location = bq.calls[0]
    assert "top_k => 20" in sql and location == "asia-south1"
    params = {p.name: p for p in job_config.query_parameters}
    assert params["max_price"].value == 20000
    assert params["query_embedding"].values == [0.6, 0.8]
    assert params["query_embedding"].array_type == "FLOAT64"


def test_query_vector_text_only_and_nothing():
    search, _, embedded = make_search([])
    search.query_vector(text="jute rug")
    assert len(embedded[0].parts) == 1 and embedded[0].parts[0].text == "jute rug"
    with pytest.raises(ValueError):
        search.query_vector()
