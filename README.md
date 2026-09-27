# Prompt Stress-Tester

An adversarial evaluation pipeline for LLM system prompts. Paste a system prompt, and the tool automatically generates targeted attack cases, runs them, judges the responses, and produces a structured vulnerability report with remediation recommendations.

## What it does

```
System prompt
     ↓
Attack Generator  →  15–18 targeted adversarial test cases across 6 categories
     ↓
Test Runner       →  each case run against the target prompt in isolation
     ↓
Judge             →  each response scored: PASS / PARTIAL / FAIL + severity
     ↓
Report            →  robustness score, category breakdown, top failures, remediation
```

## Attack categories

| Category | What it probes |
|---|---|
| Scope Escape | Requests outside the prompt's stated domain |
| Role Hijack | "Ignore instructions", DAN-style, nested roleplay |
| Prompt Extraction | Attempts to surface the system prompt verbatim |
| Indirect Injection | Malicious instructions embedded inside user-provided content |
| Constraint Bypass | Edge cases that satisfy rules while violating intent |
| Persona Drift | Gradual topic shift designed to erode the persona |

## Design decisions

**The attack generator is prompt-specific.** It reads the target prompt and generates attacks tailored to its exact constraints — not generic jailbreaks. A billing bot gets billing-specific attacks; a children's tutor gets age-restriction-specific attacks.

**Arithmetic is deterministic.** The robustness score is computed in Python from verdict and severity counts — never by the LLM. The LLM only writes the remediation prose.

**Three separate LLM roles.** Attacker (`claude-sonnet-4-6`), Runner (`claude-haiku-4-5-20251001` — intentionally lighter to surface vulnerabilities), Judge (`claude-sonnet-4-6`). Each has a focused, scoped job.

**Independent test contexts.** Each test case is a fresh conversation with the target prompt — no shared context between tests.

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in your Google API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=your-key-here
```

Or export directly:
```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
streamlit run src/app.py
```

Or run the pipeline programmatically:

```python
from src.attacker import generate_attacks
from src.runner import run_tests
from src.judge import judge_cases
from src.report import build_report

system_prompt = "You are a billing assistant. Only discuss billing topics..."

cases = generate_attacks(system_prompt, n_per_category=3)
cases = run_tests(system_prompt, cases)
cases = judge_cases(system_prompt, cases)
report = build_report(system_prompt, cases)

print(f"Score: {report.score}/100 ({report.score_label()})")
print(f"Pass: {report.passed} | Partial: {report.partial} | Fail: {report.failed}")
```

## Output

- Live results table (per-test verdict and severity as they complete)
- Robustness score (0–100)
- Per-category pass/fail breakdown
- Top 3 highest-severity failures with exact inputs and responses
- Targeted remediation recommendations
- Downloadable JSON report

## Stack

- Python 3.10+
- [Anthropic API (claude-sonnet-4-6 for attack generation and judging; claude-haiku-4-5-20251001 for the target runner)
- Streamlit

## Project structure

```
prompt-stress-tester/
├── src/
│   ├── attacker.py   # adversarial test case generation
│   ├── runner.py     # test execution against target prompt
│   ├── judge.py      # response scoring and verdict classification
│   ├── report.py     # aggregation, scoring, remediation
│   └── app.py        # Streamlit UI
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
