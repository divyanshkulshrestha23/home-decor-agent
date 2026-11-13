"""Extract the catalog's main images from ABO's small-image archive.

The archive (abo-images-small.tar, about 3 GB) holds every ABO image. This
reads it as a stream, so it can be piped straight from the download
without saving the whole archive:

    curl -s https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar \
      | python3 -m catalog.fetch_images

Only the images named in data/products.jsonl (main_image_path) are written,
to data/images/<path>.
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path


def wanted_paths(products_file):
    """Return the set of main_image_path values in the catalog."""
    paths = set()
    with Path(products_file).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                path = json.loads(line).get("main_image_path")
                if path:
                    paths.add(path)
    return paths


def relative_path(member_name, depth):
    """Last `depth` segments of an archive path, e.g. 'images/small/8c/x.jpg' -> '8c/x.jpg'."""
    parts = member_name.split("/")
    return "/".join(parts[-depth:]) if len(parts) >= depth else None


def extract(stream, wanted, out_dir):
    """Write wanted images from a tar stream into out_dir. Returns the set written."""
    out_dir = Path(out_dir)
    depth = max(len(p.split("/")) for p in wanted) if wanted else 2
    written = set()
    with tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = relative_path(member.name, depth)
            if rel not in wanted or rel in written:
                continue
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member).read()
            target.write_bytes(data)
            written.add(rel)
            if len(written) % 1000 == 0:
                print(f"  {len(written)} images written", file=sys.stderr)
            if len(written) == len(wanted):
                break
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--products", default="data/products.jsonl")
    parser.add_argument("--out", default="data/images")
    parser.add_argument("--archive", help="Path to the archive (default: read from stdin)")
    args = parser.parse_args(argv)

    wanted = wanted_paths(args.products)
    print(f"Catalog needs {len(wanted)} images.", file=sys.stderr)
    if args.archive:
        with open(args.archive, "rb") as f:
            written = extract(f, wanted, args.out)
    else:
        written = extract(sys.stdin.buffer, wanted, args.out)

    missing = sorted(wanted - written)
    print(f"Wrote {len(written)} images to {args.out}. Missing: {len(missing)}.")
    for path in missing[:10]:
        print(f"  missing: {path}")


if __name__ == "__main__":
    main()
