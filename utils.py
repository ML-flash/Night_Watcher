#!/usr/bin/env python3
"""
Run the 7‑round analysis on every article_*.json in <indir>.
Saves structured output + prompt‑chain text into <outdir>.
"""

import argparse, glob, json, logging, os, pathlib
from datetime import datetime
from typing import List, Dict, Any

from analyzer import ContentAnalyzer          # just defined above
from providers import initialize_llm_provider
from utils import save_to_file                # existing helper

# ----------------------------------------------------------------------
def _load_articles(indir: str) -> List[Dict[str, Any]]:
    arts = []
    for fp in glob.glob(os.path.join(indir, "*.json")):
        try:
            arts.append(json.load(open(fp, encoding="utf‑8")))
        except Exception as e:
            logging.error(f"Bad JSON {fp}: {e}")
    return arts


def main() -> None:
    ap = argparse.ArgumentParser(description="Night_watcher – stand‑alone analysis")
    ap.add_argument("--config", default="config.json",
                    help="LLM / provider settings JSON")
    ap.add_argument("--in", dest="indir", default="data/collected",
                    help="directory with article_*.json")
    ap.add_argument("--out", dest="outdir", default="data/analyzed",
                    help="where to store analysis outputs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(message)s")

    cfg = json.load(open(args.config, encoding="utf‑8")) if os.path.exists(args.config) else {}
    llm = initialize_llm_provider(cfg) or \
          (_ for _ in ()).throw(RuntimeError("No LLM provider available."))

    analyzer = ContentAnalyzer(llm)

    arts = _load_articles(args.indir)
    if not arts:
        logging.warning(f"No articles found in {args.indir}")
        return

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for art in arts:
        slug = "".join(c for c in art.get("title", "article") if c.isalnum())[:40] or "article"
        result = analyzer.analyze(art)

        # full JSON (all rounds)
        save_to_file(result, f"{args.outdir}/{slug}_{ts}.json")

        # readable prompt‑chain for audit
        chain_txt = []
        for rd in result["prompt_chain"]:
            chain_txt.append(f"=== ROUND {rd['round']}: {rd['name']} ===\n")
            chain_txt.append("--- PROMPT ---\n")
            chain_txt.append(rd["prompt"] + "\n\n")
            chain_txt.append("--- RESPONSE ---\n")
            chain_txt.append(rd["response"] + "\n\n")
            chain_txt.append("=" * 80 + "\n\n")
        save_to_file("".join(chain_txt), f"{args.outdir}/prompt_chain_{slug}_{ts}.txt")

    logging.info(f"✓ analyzed {len(arts)} article(s) → {args.outdir}")


if __name__ == "__main__":
    main()
