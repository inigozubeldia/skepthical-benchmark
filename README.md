# Skepthical Benchmark

Benchmark and code for the NeurIPS 2026 paper **Auditing AI peer reviewers: a benchmark on real scientific papers** (Íñigo Zubeldia, Boris Bolliet, Francisco Villaescusa-Navarro, Pablo Villanueva-Domingo, Miles Cranmer).

Errors of four types (citations, mathematics, numerics, general issues) at three severity levels are injected into real arXiv preprints, and each reviewer is scored on the errors it detects and on the false positives it raises on the clean copy. This repository contains:

- `evaluation_pipeline/`: the adjudication pipeline, to score a new reviewer;
- `corpus/`: the two 15-paper corpora (astrophysics, `astro-ph` 2026-04-22; machine learning, `cs.LG` 2026-07-15), by arXiv identifier;
- `data/<arxiv_id>/`: the 357 injected errors (`manifest.json`, `v{1,2,3}/injections.json`), the adjudication log of every reviewer (`verifications/`, `fp/`), and the API cost ledger (`costs.jsonl`);
- `prompts/`: the free-form and concise structured prompts, and the Claude Code scaffolding;
- `analysis/`: scripts reproducing every table, figure and appendix result of the paper from these files.

The papers themselves are not redistributed.

## Reviewer codes

| Code | System |
| --- | --- |
| `L1`, `L2`, `L25` | Single LLM with the free-form, structured, and concise structured prompt; tag `anthropic` (Claude Opus 4.7), `openai` (GPT-5.5) or `gemini` (Gemini 3.1 Pro) |
| `C`, `CL1`, `CL25` | Claude Code with the structured, free-form, and concise structured prompt |
| `L3` | Skepthical |
| `B1`, `B2`, `B3` | Stanford Agentic Reviewer (astrophysics only), OpenReviewer-8B, PaperAudit-Bench detector (tag `deep`) |

## Scoring a new reviewer

```bash
cd evaluation_pipeline && pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=...
# reviews at data/<arxiv_id>/reviews/<code>/<copy>.md, <copy> in v0..v3
python verify.py <arxiv_id> --systems <code>                 # detection matching on v1-v3
python fp_verify.py --paper-id <arxiv_id> --system <code>    # false positives on v0
```

## Reproducing the paper

```bash
pip install numpy pandas matplotlib statsmodels
bash analysis/run_all.sh   # a few minutes, no LLM calls; writes analysis/output/
```

Each script names the table, figure or appendix it reproduces. `analysis/derived/` holds the inputs that are not part of the adjudication log: the citation-error groups (Appendix J), the module attribution (Appendix I), and per-review agent turns and wall-clock times (Section 3.1, Table 12).

## Licence

Data (`corpus/`, `data/`, `prompts/`, `analysis/derived/`): [CC-BY-4.0](LICENSE-DATA); the papers remain under their authors' arXiv licences. Code: [MIT](LICENSE-CODE).

## Citation

```bibtex
@inproceedings{zubeldia2026auditing,
  title     = {Auditing {AI} peer reviewers: a benchmark on real scientific papers},
  author    = {Zubeldia, {\'I}{\~n}igo and Bolliet, Boris and Villaescusa-Navarro, Francisco and Villanueva-Domingo, Pablo and Cranmer, Miles},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
