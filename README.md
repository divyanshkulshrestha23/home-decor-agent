# Home Decor Agent

A multi-agent shopping and analytics assistant for a furniture and home decor store,
built with Google's Agent Development Kit (ADK), Gemini, and BigQuery.

Planned modes:

| Mode | Who | Input |
|---|---|---|
| Find similar | Shopper | Photo of an item, plus a voice note ("like this, but in oak and narrower") |
| Furnish my room | Shopper | Photo of a room, plus a voice note (budget, space, what they want) |
| Store analytics | Manager | A question about sales, answered with SQL over BigQuery |

**Status:** step 1 of 5, the product catalog, is done. The agents are not built yet.

## Step 1: Product catalog

The catalog comes from the [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
dataset (see [DATA_NOTICE.md](DATA_NOTICE.md)). `catalog/prepare.py`:

- keeps only furniture and home decor product types (sofas, chairs, tables, rugs, and similar),
  with an English name and a main image;
- flattens colour, material, style, bullet points, and keywords to English text;
- converts dimensions to centimetres and weight to kilograms;
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

# Print raw listings of one type, to check the field formats
python3 -m catalog.prepare --listings abo-listings.tar --inspect 2 --type SOFA
```

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

The tests use constructed listings in the ABO format. No dataset rows are committed.
