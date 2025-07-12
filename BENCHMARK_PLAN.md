# Symbolica Benchmark v0.1 — Project Plan

## 1. Benchmark Goals

| ID | Goal | Description |
|----|------|-------------|
| **G1** | Accuracy | How often the system produces the correct verdict. |
| **G2** | Cost | Tokens consumed and latency for any LLM calls. |
| **G3** | Explainability | Presence and quality of a trace / reasoning string. |
| **G4** | Determinism | Whether identical inputs always yield identical outputs. |

The benchmark must support evaluation of:
* **Raw LLM prompting** (baseline).
* **Symbolica rules** (with optional `PROMPT()` calls).

---

## 2. Task Taxonomy

| Suite | Share | Focus |
|-------|-------|-------|
| **S₁ – Pure-Symbolic Filtering** | ≈ 30 % | Deterministic conditions (loan approval, length limits, regex, etc.). |
| **S₂ – Hybrid Reasoning** | ≈ 30 % | Deterministic logic plus a single `PROMPT()` call (sentiment, classification). |
| **S₃ – Multi-Step / Temporal** | ≈ 25 % | Requires functions like `sustained_above()` and rule-chaining. |
| **S₄ – Complex Workflow** | ≈ 15 % | Three-rule chains or backward-chaining goals (e.g., reach `approved = true`). |

**Total target size:** 200 test cases (≈ 50 per suite).

---

## 3. Data Specification

Each test case is a single YAML file:

```yaml
id: S2-014
description: "Positive sentiment should set approved true"
facts:
  feedback: "I absolutely love this product!"
expected_verdict:
  approved: true
expected_trace: |
  vip_sentiment: PROMPT(...) == "positive"
```

**Folder layout**
```
benchmark/
  ├─ s1_symbolic/
  │    └─ *.yaml
  ├─ s2_hybrid/
  ├─ s3_temporal/
  └─ s4_workflow/
```

Reasons for YAML:
* Human-readable & diff-friendly.
* Symbolica loads YAML natively; LLM baseline runner can convert to prompts.

---

## 4. Annotation & Quality Process

1. **Draft rules** for each suite.
2. **Generate candidate inputs** (synthetic or semi-synthetic).
3. **Auto-verify** with Symbolica; ensure `expected_verdict` matches.
4. **Manual QA** 100 % of cases; tweak wording so GPT-4 can succeed but not trivially.
5. **Lock** files. PR checklist:
   * [ ] Deterministic verdict verified
   * [ ] Edge cases considered
   * [ ] Plain-English description present
   * [ ] No personally-identifiable/licensed content

---

## 5. Reproducible Evaluation Harness

```
repository/
  benchmark/
  harness/
    run_eval.py
    baselines/
       symbolica_runner.py
       gpt_runner.py
  Dockerfile
  requirements.txt
```

**Key components**
* `symbolica_runner.py` – loads rules, executes each YAML case, logs verdict + trace + exec time.
* `gpt_runner.py` – uses OpenAI client with pinned model (e.g., `gpt-4o-2024-07-10`), `temperature=0`.
* `run_eval.py` – orchestrates both runners, writes a CSV with metrics **G1-G4** & token cost.

**Reproducibility guarantees**
* Version-pin packages in `requirements.txt`.
* Provide a Dockerfile that installs exact versions and expects `OPENAI_API_KEY`.
* Include `pytest` tests so CI asserts Symbolica passes 100 % of benchmark cases.

---

## 6. Metrics & Reporting

Per-case fields captured:
`id`, `verdict_correct`, `trace_exists`, `token_input`, `token_output`, `latency_ms`.

Auto-generated aggregate table example:

| Suite | System | Accuracy | Avg tokens | Avg latency | Trace % | Cost ($) |
|-------|--------|----------|-----------|-------------|---------|----------|

Token-to-$ conversion lives in `pricing.py` (OpenAI pricing constants).

---

## 7. Licensing & Availability

* Release benchmark under **CC-BY-4.0**.
* Store results artefacts (CSVs, Markdown reports) under `releases/benchmark-v0.1/`.
* Submit to [Papers with Code](https://paperswithcode.com) after freeze.

---

## 8. Milestone Timeline

| Week | Deliverable |
|------|-------------|
| 1 | Finalise taxonomy, spec, harness scaffold. |
| 2-3 | Draft **S₁ & S₂** rules + 100 cases; manual QA. |
| 4 | Add **S₃** temporal harness functions; 50 cases. |
| 5 | Add **S₄** workflow cases; backward-chaining eval. |
| 6 | Freeze v0.1, run GPT-4 & Symbolica baselines. |
| 7 | Public release, blog post, community outreach. |

---

## 9. Immediate Next Actions

1. Create `benchmark/` and `harness/` skeleton in a feature branch.
2. Seed **S₁** with rules from existing examples.
3. Write `run_eval.py` CLI stub that prints “hello benchmark”.
4. Open three “good first issue” tickets:
   * Add one **S₁** test case.
   * Add one **S₂** test case.
   * Improve README for benchmark spec.

By following this structured plan we transform benchmark creation into a series of tangible pull-requests, making it accessible for contributors and providing the data needed for the upcoming Benchmark paper. 

---

## 10. Updated Next Actions (after harness scaffold)

| Status | Task |
|--------|------|
| ✅ | Scaffold `benchmark/` suites and seed six YAML cases |
| ✅ | Add canonical `credit_fraud_rules.yaml` rules file |
| ✅ | Implement `SymbolicaRunner` with deterministic stub `PROMPT()` |
| ✅ | Implement CLI harness `harness/run_eval.py` (summary + CSV) |

**Upcoming (v0.1)**
1. **GPTRunner implementation** – real OpenAI calls, token/latency logging, cost calc (opt-in via `OPENAI_API_KEY`).
2. **CI integration** – `pytest` job that executes `run_eval.py` and asserts 100 % accuracy for Symbolica baseline.
3. **Metric enrichment** – add token counts & deterministic seed for latency benchmarking.
4. **Dataset growth** – expand to 25 test cases (S₁ × 8, S₂ × 8, S₃ × 5, S₄ × 4) via community PRs.
5. **Docs update** – create `benchmark/README.md` with run instructions and contribution guide.
6. **Dockerfile** – pin versions and embed harness CLI for reproducibility.
7. **GitHub issues** – create “good first issue” tickets for adding test cases and GPTRunner enhancements.

Progress will be tracked in the project board `benchmark-v0.1`. When all upcoming items are ✅ we freeze v0.1 and publish baseline results. 