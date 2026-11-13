"""Try the find-similar agent from the command line.

Usage (Cloud Shell):
    python3 -m agent.cli --text "What was revenue in October 2026?"      # routed to the analyst
    python3 -m agent.cli --image chair.jpg --text "like this but in navy, under 15k"
    python3 -m agent.cli --image chair.jpg --audio note.m4a
    python3 -m agent.cli --text "a jute rug for a small bedroom"

Prints the reply, then each tool call: products found, or the SQL result rows.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif"}
AUDIO_TYPES = {".m4a": "audio/m4a", ".mp3": "audio/mp3", ".wav": "audio/wav", ".webm": "audio/webm",
               ".ogg": "audio/ogg", ".aac": "audio/x-aac", ".flac": "audio/flac", ".mp4": "audio/mp4"}


def read_file(path, types_by_suffix, kind):
    suffix = Path(path).suffix.lower()
    if suffix not in types_by_suffix:
        sys.exit(f"Unsupported {kind} type {suffix}. Use one of: {', '.join(sorted(types_by_suffix))}")
    return Path(path).read_bytes(), types_by_suffix[suffix]


def configure_vertex(project, location):
    """ADK and google-genai read these to call Gemini through Vertex AI."""
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location


def print_result(result):
    print("\n=== Agent reply ===\n" + (result["reply"] or "(no reply)"))
    for call in result["tool_calls"]:
        print(f"\n=== Tool call: {call['name']} ===")
        print(f"args: {call['args']}")
        response = call.get("response") or {}
        if not isinstance(response, dict):
            print(f"result: {str(response)[:300]}")
            continue
        if "error" in response:
            print(f"error: {response['error']}")
        if "rows" in response:
            for row in response["rows"][:10]:
                print(f"  {row}")
            if response["row_count"] > 10 or response.get("truncated"):
                print(f"  ... {response['row_count']} rows" + (" (truncated)" if response.get("truncated") else ""))
        for i, product in enumerate(response.get("results", []), start=1):
            print(f"  {i}. [{product['product_id']}] {product['name'][:70]}")
            print(f"     Rs {product['price_inr']:,} | {product['category']} | "
                  f"{', '.join(product['colors']) or '-'} | {product['size_cm_w_d_h']} cm | "
                  f"similarity {product['similarity']}")


def build_agent(kind, project, model):
    """kind: 'all' (router with both agents), 'find_similar' or 'analyst'."""
    from google.cloud import bigquery

    from agent import config
    from agent.agent import build_find_similar_agent
    from agent.analyst import build_analyst_agent
    from agent.router import build_root_agent
    from agent.search import ProductSearch
    from agent.sql_tools import SqlRunner

    search = ProductSearch.from_environment(project, config.DATASET, config.BQ_LOCATION, config.MODEL_LOCATION)
    runner = SqlRunner(bigquery.Client(project=project), project, config.DATASET, config.BQ_LOCATION)
    if kind == "find_similar":
        return build_find_similar_agent(search, model=model)
    if kind == "analyst":
        return build_analyst_agent(runner, model=model)
    return build_root_agent(search, runner, model=model)


async def main_async(args):
    from agent.run import AgentSession, build_message

    session = AgentSession(build_agent(args.agent, args.project, args.model))
    session_id = await session.new_session()
    message = build_message(
        text=args.text,
        image=read_file(args.image, IMAGE_TYPES, "image") if args.image else None,
        audio=read_file(args.audio, AUDIO_TYPES, "audio") if args.audio else None,
    )
    print_result(await session.send(session_id, message))


def main(argv=None):
    from agent import config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--text")
    parser.add_argument("--project", default=config.PROJECT or None)
    parser.add_argument("--model", default=config.AGENT_MODEL)
    parser.add_argument("--agent", choices=["all", "find_similar", "analyst"], default="all",
                        help="all = the routing agent in front of both (default)")
    args = parser.parse_args(argv)
    if not args.project:
        parser.error("--project is required (or set GOOGLE_CLOUD_PROJECT)")
    if not (args.image or args.audio or args.text):
        parser.error("give at least one of --image, --audio, --text")
    configure_vertex(args.project, config.MODEL_LOCATION)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
