import io
import json
import tarfile

import pytest

from catalog import embed_images as emb
from catalog import fetch_images


def write_products(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def product(pid, category="chair", image=None, **extra):
    row = {
        "product_id": pid, "name": f"{category.title()} {pid}", "category": category,
        "colors": ["Navy"], "materials": ["Oak", "Linen", "Steel", "Extra"], "styles": [],
        "main_image_path": image if image is not None else f"aa/{pid}.jpg",
    }
    row.update(extra)
    return row


# --- fetch_images -----------------------------------------------------------

def make_tar(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_extract_writes_only_wanted_images(tmp_path):
    stream = make_tar({
        "images/small/aa/one.jpg": b"1",
        "images/small/bb/two.jpg": b"2",
        "images/small/cc/other.jpg": b"x",
        "images/metadata/images.csv.gz": b"m",
    })
    written = fetch_images.extract(stream, {"aa/one.jpg", "bb/two.jpg", "dd/gone.jpg"}, tmp_path)
    assert written == {"aa/one.jpg", "bb/two.jpg"}
    assert (tmp_path / "aa/one.jpg").read_bytes() == b"1"
    assert not (tmp_path / "cc/other.jpg").exists()


def test_fetch_images_main_reports_missing(tmp_path, capsys):
    products = write_products(tmp_path / "p.jsonl", [product("A", image="aa/one.jpg"), product("B", image="zz/no.jpg")])
    archive = tmp_path / "small.tar"
    archive.write_bytes(make_tar({"images/small/aa/one.jpg": b"1"}).getvalue())
    fetch_images.main(["--products", str(products), "--out", str(tmp_path / "img"), "--archive", str(archive)])
    out = capsys.readouterr().out
    assert "Wrote 1 images" in out and "Missing: 1." in out and "zz/no.jpg" in out


# --- embedding helpers ------------------------------------------------------

def test_product_text_is_short_and_descriptive():
    text = emb.product_text(product("A", category="accent_furniture", styles=["Modern"]))
    assert text == "Accent_Furniture A. Category: accent furniture. Colour: Navy. Material: Oak, Linen, Steel. Style: Modern"
    long = emb.product_text(product("B", name="x" * 1000))
    assert len(long) == emb.MAX_TEXT_CHARS


def test_build_content_parts():
    content = emb.build_content(b"img", "image/jpeg", "a chair")
    assert len(content.parts) == 2
    assert content.parts[0].inline_data.mime_type == "image/jpeg"
    assert content.parts[1].text == "a chair"
    assert len(emb.build_content(b"img", "image/png").parts) == 1


def test_normalize():
    assert emb.normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ValueError):
        emb.normalize([0.0, 0.0])


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, config.output_dimensionality))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class Embedding:
            values = outcome

        class Response:
            embeddings = [Embedding()]

        return Response()


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def api_error(code):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


def test_embedder_retries_rate_limits_then_succeeds():
    client = FakeClient([api_error(429), api_error(503), [0.0, 2.0]])
    sleeps = []
    embed = emb.make_embedder(client, dims=2, sleep=sleeps.append)
    assert embed(object()) == [0.0, 1.0]
    assert sleeps == [1, 2]
    assert client.models.calls[0] == ("gemini-embedding-2", 2)


def test_embedder_does_not_retry_client_errors():
    embed = emb.make_embedder(FakeClient([api_error(403)]), sleep=lambda s: None)
    with pytest.raises(Exception) as info:
        embed(object())
    assert info.value.code == 403


# --- embed_catalog ----------------------------------------------------------

@pytest.fixture
def catalog(tmp_path):
    rows = [product(p) for p in "ABCD"] + [product("E", image="aa/missing.jpg")]
    images = tmp_path / "images"
    (images / "aa").mkdir(parents=True)
    for p in "ABCD":
        (images / "aa" / f"{p}.jpg").write_bytes(b"jpg")
    return rows, images, tmp_path / "embeddings.jsonl"


def test_embed_catalog_writes_and_resumes(catalog):
    rows, images, out = catalog
    seen = []

    def embed(content):
        seen.append(len(content.parts))
        return [1.0, 0.0]

    first = emb.embed_catalog(rows, images, embed, out, limit=2, workers=1)
    assert (first["written"], first["skipped_no_image"], first["failed"]) == (2, 1, 0)
    second = emb.embed_catalog(rows, images, embed, out, workers=2)
    assert (second["already_done"], second["written"]) == (2, 2)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["product_id"] for r in records) == ["A", "B", "C", "D"]
    assert records[0]["content_mode"] == "image_text" and records[0]["dims"] == 768
    assert set(seen) == {2}


def test_embed_catalog_image_only_mode(catalog):
    rows, images, out = catalog
    seen = []
    emb.embed_catalog(rows, images, lambda c: seen.append(len(c.parts)) or [1.0], out, mode="image", workers=1)
    assert set(seen) == {1}


def test_embed_catalog_stops_when_every_call_fails(catalog, tmp_path):
    rows, images, out = catalog
    rows = rows + [product(f"X{i}") for i in range(20)]
    for i in range(20):
        (images / "aa" / f"X{i}.jpg").write_bytes(b"jpg")

    def embed(content):
        raise PermissionError("no access to model")

    summary = emb.embed_catalog(rows, images, embed, out, workers=1)
    assert summary["written"] == 0
    assert summary["failed"] == emb.STOP_AFTER_FAILURES
    assert "PermissionError: no access to model" in summary["first_errors"][0]


def test_nearest_neighbour_check(tmp_path):
    products = write_products(tmp_path / "p.jsonl", [
        product("A", "chair"), product("B", "chair"), product("C", "rug"), product("D", "rug"),
    ])
    vectors = {"A": [1, 0, 0], "B": [0.9, 0.1, 0], "C": [0, 1, 0], "D": [0.8, 0.6, 0]}
    (tmp_path / "e.jsonl").write_text("".join(
        json.dumps({"product_id": k, "embedding": emb.normalize(v)}) + "\n" for k, v in vectors.items()
    ))
    result = emb.nearest_neighbour_check(tmp_path / "e.jsonl", products)
    # A->B (chair), B->A (chair), C->D (rug), D->B (chair, wrong): 3 of 4
    assert result["products"] == 4
    assert result["same_category_top1"] == 75.0
    assert result["examples"][0][:2] == ("Chair A", "Chair B")


def test_cli_requires_project_unless_checking(tmp_path):
    with pytest.raises(SystemExit):
        emb.main(["--products", str(tmp_path / "x")])
