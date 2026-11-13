# Home Decor Agent

A multi-agent shopping and analytics assistant for a furniture and home decor store,
built with Google's Agent Development Kit (ADK), Gemini, and BigQuery.

Planned modes:

| Mode | Who | Input |
|---|---|---|
| Find similar | Shopper | Photo of an item, plus a voice note ("like this, but in oak and narrower") |
| Furnish my room | Shopper | Photo of a room, plus a voice note (budget, space, what they want) |
| Store analytics | Manager | A question about sales, answered with SQL over BigQuery |

**Status:** the catalog (14,224 products) and its image embeddings are in BigQuery. The
find-similar agent (step 2b) is built and tested offline; it has not been run against live
Gemini yet. Furnish-my-room and analytics are not built yet.

## Step 1: Product catalog

The catalog comes from the [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
dataset (see [DATA_NOTICE.md](DATA_NOTICE.md)). `catalog/prepare.py`:

- keeps only furniture and home decor product types (sofas, chairs, tables, rugs, and similar),
  with an English name and a main image;
- sorts products from broad types such as `HOME` into store categories by their category
  trail (falling back to words in the name), and leaves out what the store would not sell,
  such as tools, grill covers, and phone accessories (`catalog/classify.py`);
- flattens colour, material, style, bullet points, and keywords to English text;
- converts dimensions to centimetres and weight to kilograms, and adds the longest and
  shortest sides of each product's footprint (ABO's own "width" and "length" are not
  used consistently, so a sofa's "length" can be its long side);
- generates a price in rupees for each product (ABO has no prices; see `catalog/pricing.py`);
- writes `data/products.jsonl` and a coverage report to `data/catalog_report.json`.

`catalog/load_bigquery.py` checks every row against the table schema, then loads it
into the BigQuery table `home_decor.products`.

### Run it in Cloud Shell

Cloud Shell already has Python, `gcloud`, and Google Cloud credentials.

```bash
git clone https://github.com/divyanshkulshrestha23/home-decor-agent.git
cd home-decor-agent
pip install -r requirements.txt

# Download the listings (about 83 MB) and the image index
curl -O https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
curl -O https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz

# Build the catalog
python3 -m catalog.prepare --listings abo-listings.tar --images-metadata images.csv.gz

# Check the rows, then load them
gcloud services enable bigquery.googleapis.com
python3 -m catalog.load_bigquery --project YOUR_PROJECT_ID --dry-run
python3 -m catalog.load_bigquery --project YOUR_PROJECT_ID
```

`--images-metadata` is optional. Without it, `main_image_path` is left empty.

To run on your own machine instead, use `python` in place of `python3` on Windows, and
run `gcloud auth application-default login` before the load step.

### Other options

```bash
# List every product type in the dataset, and which ones the store keeps
python3 -m catalog.prepare --listings abo-listings.tar --list-types

# Show the most common category trails for some product types
python3 -m catalog.prepare --listings abo-listings.tar --browse-paths HOME HOME_BED_AND_BATH

# Print raw listings of one type, to check the field formats
python3 -m catalog.prepare --listings abo-listings.tar --inspect 2 --type SOFA
```

## Step 2a: Image embeddings

Each product gets one vector from **Gemini Embedding 2** (`gemini-embedding-2`, 768 dimensions),
built from its main image plus a short line of text (name, category, colour, material, style).
A shopper's photo plus spoken details can then be embedded the same way and matched against it.

```bash
git pull && pip install -r requirements.txt
gcloud services enable aiplatform.googleapis.com

# 1. Extract the catalog's images (streams the ~3 GB archive; keeps only ours)
curl -s https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar \
  | python3 -m catalog.fetch_images

# 2. Test on 100 products, then sanity-check the vectors
python3 -m catalog.embed_images --project YOUR_PROJECT_ID --limit 100
python3 -m catalog.embed_images --check

# 3. Embed the rest (resumes where it stopped), then load into BigQuery
python3 -m catalog.embed_images --project YOUR_PROJECT_ID
python3 -m catalog.embed_images --project YOUR_PROJECT_ID --load
```

`--check` reports how often each product's closest neighbour is in the same category. It is a
rough check that the vectors mean something, not a search accuracy score.

## Step 2b: Find-similar agent

An ADK agent (`agent/agent.py`, model `gemini-3.5-flash-lite` by default) reads the shopper's
photo, voice note and text, turns any firm limits into filters, and calls one tool,
`find_similar_products` (`agent/tools.py`). The tool:

1. embeds the photo plus the agent's description of what the shopper wants, in the same
   image-plus-text format as the catalog;
2. runs one BigQuery query that ranks the products passing the filters (category, price,
   colour, material, width, depth, height) by cosine similarity with `ML.DISTANCE`. This is
   an exact search; at 14k products it needs no vector index. (`VECTOR_SEARCH` was tried
   first, but it does not accept a filtered input that joins two tables.) Filters are query
   parameters, never pasted into the SQL;
3. drops near-duplicate colour and size variants and returns the top 5.

Try it from Cloud Shell (upload a photo or a voice note with the Cloud Shell "Upload" menu):

```bash
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
python3 -m agent.cli --text "a jute rug for a small bedroom, under 5000"
python3 -m agent.cli --image my_chair.jpg --text "like this but in navy"
python3 -m agent.cli --image my_chair.jpg --audio note.m4a
```

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

The tests use constructed listings in the ABO format. No dataset rows are committed.
