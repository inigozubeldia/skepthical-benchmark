"""Shared paths and loaders for the analysis scripts. Reads only corpus/ and data/; no LLM calls."""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUT = REPO / "analysis" / "output"

CORPORA = {
    "astro": (REPO / "corpus" / "astro_15.txt").read_text().split(),
    "ml": (REPO / "corpus" / "ml_15.txt").read_text().split(),
}
CORPUS_NAMES = {"astro": "Astrophysics corpus", "ml": "Machine-learning corpus"}

MODS = ("citations", "numerics", "mathematics", "general")
PROVIDERS = ("anthropic", "openai", "gemini")
PROVIDER_MODEL = {"anthropic": "Claude Opus 4.7", "openai": "GPT-5.5", "gemini": "Gemini 3.1 Pro"}

# Table 2 systems in table order: (folder under verifications/, file tag, display name).
SYSTEMS = [
    ("L1", "anthropic", "Claude Opus 4.7 (free-form)"),
    ("L1", "openai", "GPT-5.5 (free-form)"),
    ("L1", "gemini", "Gemini 3.1 Pro (free-form)"),
    ("L2", "anthropic", "Claude Opus 4.7 (structured)"),
    ("L2", "openai", "GPT-5.5 (structured)"),
    ("L2", "gemini", "Gemini 3.1 Pro (structured)"),
    ("L3", "", "Skepthical"),
    ("C", "", "Claude Code (structured)"),
    ("CL1", "", "Claude Code (free-form)"),
    ("B1", "", "Stanford Agentic Reviewer"),
    ("B2", "", "OpenReviewer-8B"),
    ("B3", "deep", "PaperAudit-Bench Deep"),
]
CONCISE = [
    ("L25", "anthropic", "Claude Opus 4.7 (concise structured)"),
    ("L25", "openai", "GPT-5.5 (concise structured)"),
    ("L25", "gemini", "Gemini 3.1 Pro (concise structured)"),
    ("CL25", "", "Claude Code (concise structured)"),
]

VER_RE = re.compile(r"^(v[1-3])(?:__(.+))?$")
FP_RE = re.compile(r"^([A-Z][A-Za-z0-9]*)(?:__(.+?))?_v0$")


def corpus(name: str) -> list[str]:
    return CORPORA[name]


def load_injections(papers):
    """{(paper, copy, module): severity} for every injected error."""
    inj = {}
    for p in papers:
        for c in json.loads((DATA / p / "manifest.json").read_text()).get("copies", []):
            for i in c.get("injections", []):
                inj[(p, c["copy_id"], i["module"])] = i["severity"]
    return inj


def iter_verdicts(papers):
    """Yield (system, tag, paper, copy, verdict_record) for every adjudicated injection."""
    for p in papers:
        for f in sorted(glob.glob(str(DATA / p / "verifications" / "*" / "*.json"))):
            sysd = Path(f).parent.name
            m = VER_RE.match(Path(f).stem)
            if not m:
                continue
            tag = (m.group(2) or "").lower()
            for v in json.loads(Path(f).read_text()).get("verdicts", []):
                yield sysd, tag, p, m.group(1), v


def load_detections(papers):
    """{(system, tag): {(paper, copy, module): detected}}."""
    inj = load_injections(papers)
    D = defaultdict(dict)
    for sysd, tag, p, copy, v in iter_verdicts(papers):
        k = (p, copy, v.get("module"))
        if k in inj:
            D[(sysd, tag)][k] = bool(v.get("detected"))
    return D, inj


def load_fp(papers):
    """{(system, tag): {module: [(paper, verdict), ...]}} from the clean-copy adjudications."""
    F = defaultdict(lambda: defaultdict(list))
    for p in papers:
        for f in sorted(glob.glob(str(DATA / p / "fp" / "*_v0.json"))):
            m = FP_RE.match(Path(f).stem)
            if not m:
                continue
            s, t = m.group(1), (m.group(2) or "").lower()
            for mod, vs in (json.loads(Path(f).read_text()).get("per_module_verdicts") or {}).items():
                for v in vs:
                    F[(s, t)][mod].append((p, v.get("verdict")))
    return F
