"""Find products similar to a photo and/or description.

The query (photo, text, or both) is embedded with the same model and the
same image-plus-text format as the catalog, then one BigQuery query ranks
the products that pass the filters by cosine similarity (ML.DISTANCE). Filters are
applied before the search, so "under Rs 20,000" never returns a Rs 40,000
sofa that happened to look similar.

All filter values are passed as query parameters, never pasted into SQL,
because they come from a language model reading what the shopper said.
"""

from dataclasses import dataclass, field

from catalog.embed_images import build_content, make_embedder

SEARCH_POOL = 4  # fetch this many times top_k, then drop near-duplicate variants


@dataclass
class Filters:
    categories: list[str] = field(default_factory=list)
    min_price_inr: int | None = None
    max_price_inr: int | None = None
    colors: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    max_width_cm: float | None = None   # longest side of the footprint
    max_depth_cm: float | None = None   # shortest side of the footprint
    max_height_cm: float | None = None
    exclude_ids: list[str] = field(default_factory=list)

    def describe(self):
        """The filters that are set, for showing back to the agent."""
        return {k: v for k, v in self.__dict__.items() if v not in (None, []) and k != "exclude_ids"}


def _matches_any(column_expr, param):
    """True if, for at least one wanted phrase, every word of it appears in the product's values.

    The product's values (e.g. colour names plus standard colour families) are
    joined into one lowercase string. "navy blue" then matches a product listed
    as colour "Navy" in the "blue" family, which an exact-phrase match misses.
    STRPOS is used rather than a regex so the shopper's words are never
    interpreted as a pattern.
    """
    return (
        f"EXISTS (SELECT 1 FROM UNNEST(@{param}) AS want WHERE NOT EXISTS ("
        f"SELECT 1 FROM UNNEST(SPLIT(LOWER(TRIM(want)), ' ')) AS word "
        f"WHERE word != '' AND STRPOS(LOWER(ARRAY_TO_STRING({column_expr}, ' ')), word) = 0))"
    )


def build_query(filters, project, dataset, top_k):
    """Return (sql, params). params is a list of (name, bigquery_type, value).

    bigquery_type is 'INT64', 'FLOAT64', 'STRING', 'ARRAY<STRING>' or
    'ARRAY<FLOAT64>'. The query embedding is added by the caller as @query_embedding.
    """
    if not isinstance(top_k, int) or not 1 <= top_k <= 200:
        raise ValueError("top_k must be an integer from 1 to 200")
    where, params = [], []

    def add(condition, name, kind, value):
        where.append(condition)
        params.append((name, kind, value))

    if filters.categories:
        add("p.category IN UNNEST(@categories)", "categories", "ARRAY<STRING>", list(filters.categories))
    if filters.min_price_inr is not None:
        add("p.price_inr >= @min_price", "min_price", "INT64", int(filters.min_price_inr))
    if filters.max_price_inr is not None:
        add("p.price_inr <= @max_price", "max_price", "INT64", int(filters.max_price_inr))
    if filters.colors:
        add(_matches_any("ARRAY_CONCAT(p.colors, p.color_families)", "colors"), "colors",
            "ARRAY<STRING>", list(filters.colors))
    if filters.materials:
        add(_matches_any("p.materials", "materials"), "materials", "ARRAY<STRING>", list(filters.materials))
    if filters.max_width_cm is not None:
        add("p.longest_side_cm <= @max_width", "max_width", "FLOAT64", float(filters.max_width_cm))
    if filters.max_depth_cm is not None:
        add("p.shortest_side_cm <= @max_depth", "max_depth", "FLOAT64", float(filters.max_depth_cm))
    if filters.max_height_cm is not None:
        add("p.height_cm <= @max_height", "max_height", "FLOAT64", float(filters.max_height_cm))
    if filters.exclude_ids:
        add("p.product_id NOT IN UNNEST(@exclude_ids)", "exclude_ids", "ARRAY<STRING>", list(filters.exclude_ids))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    table = f"`{project}.{dataset}"
    # Exact cosine search over the filtered products. VECTOR_SEARCH was tried
    # first, but it rejects a filtering subquery that joins two tables; without
    # a vector index it would do this same exact computation anyway.
    sql = f"""
SELECT p.product_id, p.name, p.brand, p.category, p.price_inr, p.colors, p.materials,
       p.styles, p.longest_side_cm, p.shortest_side_cm, p.height_cm, p.main_image_path,
       1 - ML.DISTANCE(e.embedding, @query_embedding, 'COSINE') AS similarity
FROM {table}.products` AS p
JOIN {table}.product_embeddings` AS e USING (product_id)
{where_sql}
ORDER BY similarity DESC
LIMIT {top_k}
""".strip()
    return sql, params


def variant_key(row):
    """Products that are colour or size variants of one item share an image or a name stem."""
    stem = row["name"].lower().split(",")[0].strip()[:50]
    return row.get("main_image_path") or stem, stem


def drop_variants(rows, limit):
    """Keep the best-ranked product of each variant group, up to limit."""
    seen_images, seen_stems, kept = set(), set(), []
    for row in rows:
        image, stem = variant_key(row)
        if image in seen_images or stem in seen_stems:
            continue
        seen_images.add(image)
        seen_stems.add(stem)
        kept.append(row)
        if len(kept) == limit:
            break
    return kept


def format_product(row):
    """Compact product summary for the agent and the CLI."""
    size = [f"{row[k]:.0f}" if row.get(k) else "?" for k in ("longest_side_cm", "shortest_side_cm", "height_cm")]
    return {
        "product_id": row["product_id"],
        "name": row["name"],
        "brand": row.get("brand"),
        "category": row["category"],
        "price_inr": row["price_inr"],
        "colors": list(row.get("colors") or [])[:3],
        "materials": list(row.get("materials") or [])[:3],
        "size_cm_w_d_h": " x ".join(size),
        "similarity": round(float(row["similarity"]), 3),
        "image_path": row.get("main_image_path"),
    }


class ProductSearch:
    """Embeds a query and searches BigQuery. Clients are injected so tests can fake them."""

    def __init__(self, bq_client, embed, project, dataset, bq_location):
        self.bq = bq_client
        self.embed = embed
        self.project = project
        self.dataset = dataset
        self.bq_location = bq_location

    @classmethod
    def from_environment(cls, project, dataset, bq_location, model_location):
        from google import genai
        from google.cloud import bigquery

        genai_client = genai.Client(vertexai=True, project=project, location=model_location)
        return cls(bigquery.Client(project=project), make_embedder(genai_client), project, dataset, bq_location)

    def query_vector(self, image_bytes=None, image_mime=None, text=None):
        if image_bytes:
            content = build_content(image_bytes, image_mime or "image/jpeg", text or None)
        elif text:
            from google.genai import types

            content = types.Content(parts=[types.Part.from_text(text=text)])
        else:
            raise ValueError("A photo or a description is needed to search.")
        return self.embed(content)

    def run_sql(self, sql, params, vector):
        from google.cloud import bigquery

        def to_param(name, kind, value):
            if kind.startswith("ARRAY<"):
                return bigquery.ArrayQueryParameter(name, kind[6:-1], value)
            return bigquery.ScalarQueryParameter(name, kind, value)

        query_params = [to_param(*p) for p in params]
        query_params.append(bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", vector))
        job = self.bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=query_params),
            location=self.bq_location,
        )
        return [dict(row.items()) for row in job.result()]

    def search(self, filters, image_bytes=None, image_mime=None, text=None, top_k=5):
        vector = self.query_vector(image_bytes, image_mime, text)
        sql, params = build_query(filters, self.project, self.dataset, top_k * SEARCH_POOL)
        rows = self.run_sql(sql, params, vector)
        return [format_product(r) for r in drop_variants(rows, top_k)]
