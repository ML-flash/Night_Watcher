#!/usr/bin/env python3
"""
Run the 7‑round Night_watcher analysis as a stand‑alone step.
Keeps LM‑Studio → Anthropic fallback logic unchanged.
"""

import argparse, glob, json, logging, os, pathlib, sys
from datetime import datetime
from typing import Any, Dict, List

# ── our own helpers / provider -------------------------------------------------
from providers import initialize_llm_provider       # existing file in repo
from analyzer  import ContentAnalyzer               # 7‑round chain (unchanged)
from utils     import save_to_file                  # existing helper

# ── minimal, built‑in config loader (replaces import night_watcher) ------------
DEFAULT_CFG: Dict[str, Any] = {
    "llm_provider": {
        "type": "lm_studio",
        "host": "http://localhost:1234",
        "model": "default"
    }
}

def load_config(path: str) -> Dict[str, Any]:
    """Return JSON config or fallback to DEFAULT_CFG."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf‑8") as f:
                cfg = json.load(f)
            # shallow‑merge defaults so those keys are always present
            merged = DEFAULT_CFG | cfg
            merged["llm_provider"] = DEFAULT_CFG["llm_provider"] | cfg.get("llm_provider", {})
            return merged
        except Exception as e:
            logging.warning(f"Bad config JSON – using defaults ({e})")
    return DEFAULT_CFG

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
    ap = argparse.ArgumentParser(description="Night_watcher stand‑alone analysis")
    ap.add_argument("--config", default="config.json",
                    help="Optional config JSON (falls back to defaults)")
    ap.add_argument("--in", dest="indir", default="data/collected",
                    help="directory with article_*.json")
    ap.add_argument("--out", dest="outdir", default="data/analyzed",
                    help="directory to write analysis outputs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(message)s")

    # ► 1. load config (adds default LM‑Studio details if file missing)
    cfg = load_config(args.config)

    # ► 2. get provider – this will:
    #    – ping LM Studio   → use if reachable
    #    – otherwise prompt → paste Anthropic key + model
    llm = initialize_llm_provider(cfg)
    if llm is None:
        logging.error("No LLM provider available. Aborting.")
        return

    analyzer = ContentAnalyzer(llm)

    # ► 3. read articles
    arts = _load_articles(args.indir)
    if not arts:
        logging.warning(f"No articles found in {args.indir}")
        return

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ► 4. run analysis
    for art in arts:
        slug = "".join(c for c in art.get("title", "article") if c.isalnum())[:40] or "article"
        result = analyzer.analyze(art)

        save_to_file(result, f"{args.outdir}/{slug}_{ts}.json")

        # human‑readable prompt chain (for auditing)
        chain_txt = []
        for rd in result["prompt_chain"]:
            chain_txt.append(f"=== ROUND {rd['round']}: {rd['name']} ===\n")
            chain_txt.append("--- PROMPT ---\n" + rd["prompt"] + "\n\n")
            chain_txt.append("--- RESPONSE ---\n" + rd["response"] + "\n\n")
            chain_txt.append("=" * 80 + "\n\n")
        save_to_file("".join(chain_txt),
                     f"{args.outdir}/prompt_chain_{slug}_{ts}.txt")

    logging.info(f"✓ analyzed {len(arts)} article(s) → {args.outdir}")


if __name__ == "__main__":
    main()
