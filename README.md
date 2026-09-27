# Prompt Stress-Tester

An adversarial evaluation pipeline for LLM system prompts. Paste a system prompt and get two scores: a **Prompt Quality Score** (what the prompt says) and a **Robustness Score** (how the model behaves under attack) — plus a gap analysis that tells you whether your base model is compensating for missing prompt-level defenses.

## The core insight

Most LLM eval tools only measure model behavior. A strong base model can score 92/100 on robustness even with a near-empty prompt — because the model's own safety training does the work the prompt didn't. That gap is a production risk: a different model, a fine-tuned variant, or a future update may not hold.

This tool separates the two:

```
Prompt Quality Score  — what the prompt explicitly defends against
Robustness Score      — how the model actually behaved under 18 targeted attacks
Gap Analysis          — "your prompt scored 35 but your model scored 92;
                         the base model is compensating for these missing defenses"
```

## Pipeline

```
System prompt in
       ↓
Prompt Quality Analyzer  →  scores the prompt on 8 structural dimensions
       ↓
Attack Generator         →  18 targeted adversarial test cases (6 categories)
       ↓
Test Runner              →  each case sent to the target prompt in isolation
       ↓
Judge                    →  each response scored: PASS / PARTIAL / FAIL + severity
       ↓
Report                   →  dual score, gap analysis, category breakdown,
                            top failures, remediation recommendations
```

## Prompt Quality dimensions (8-point rubric)

| Dimension | What it checks | Weight |
|---|---|---|
| Confidentiality Instruction | Does the prompt tell the model to keep itself secret? | 20% |
| Scope Definition | Are in-scope and out-of-scope topics explicitly defined? | 15% |
| Injection Guards | Does it anticipate instructions embedded in user-provided content? | 15% |
| Constraint Specificity | Are constraints hard rules or vague suggestions? | 12% |
| Persona Lock | Is the identity anchored and forbidden from drifting? | 12% |
| Escalation Path | Is there a defined fallback when the model can't help? | 10% |
| Bypass Anticipation | Does it anticipate hypothetical/fictional/roleplay framings? | 8% |
| Role Hijack Resistance | Does it explicitly forbid identity override attempts? | 8% |

## Attack categories (6-category adversarial suite)

| Category | What it probes |
|---|---|
| Scope Escape | Requests outside the prompt's stated domain |
| Role Hijack | "Ignore instructions", DAN-style, nested roleplay |
| Prompt Extraction | Attempts to surface the system prompt verbatim or paraphrased |
| Indirect Injection | Malicious instructions embedded inside user-provided content |
| Constraint Bypass | Edge cases that satisfy rules while violating intent |
| Persona Drift | Gradual topic shift designed to erode the persona |

## Design decisions

**Two scores, not one.** Prompt quality is evaluated by reading the prompt directly. Robustness is evaluated by attacking it. They are independent — a well-written prompt on a weak model can score differently from a weak prompt on a strong model. The gap between them is the most actionable output.

**Arithmetic is deterministic.** Both scores are computed in Python from structured LLM outputs — never by free-form LLM judgment on a number. The LLM judges verdicts and dimension presence; the math is in code.

**Three separate LLM roles.** Attacker (`claude-sonnet-4-6`), Runner (`claude-haiku-4-5` — intentionally lighter to surface vulnerabilities), Judge (`claude-sonnet-4-6`). Each has a focused, scoped job.

**Attacks are prompt-specific.** The generator reads the target prompt and crafts attacks against its specific constraints — not generic jailbreaks. A billing bot gets billing-specific attacks; a children's tutor gets age-restriction-specific attacks.

**Independent test contexts.** Each test case is a fresh conversation with no shared context — results aren't contaminated by prior turns.

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in your Anthropic API key
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

Or export directly:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
streamlit run src/app.py
```

Four pre-loaded demo prompts ship with the tool, including a deliberately weak prompt (ShopEasy) that demonstrates a large quality/robustness gap.

## Programmatic usage

```python
from src.attacker import generate_attacks
from src.runner import run_tests
from src.judge import judge_cases
from src.report import build_report
from src.analyzer import analyze_prompt

system_prompt = "You are a billing assistant. Only discuss billing topics..."

# Score the prompt itself
quality = analyze_prompt(system_prompt)
print(f"Prompt Quality: {quality.overall_score}/100 (Grade: {quality.grade})")

# Run adversarial attacks
cases = generate_attacks(system_prompt, n_per_category=3)
cases = run_tests(system_prompt, cases)
cases = judge_cases(system_prompt, cases)
report = build_report(system_prompt, cases)

print(f"Robustness: {report.score}/100 ({report.score_label()})")
print(f"Pass: {report.passed} | Partial: {report.partial} | Fail: {report.failed}")
```

## Output

- Prompt Quality Score (0–100) with grade and per-dimension breakdown
- Robustness Score (0–100) with pass/partial/fail counts
- Gap analysis: interpretation of the difference between the two scores
- Per-category robustness breakdown
- Top 3 highest-severity failures with exact inputs and responses
- Targeted remediation recommendations with specific prompt language
- Downloadable JSON report

## Stack

- Python 3.10+
- [Anthropic API](https://docs.anthropic.com) — `claude-sonnet-4-6` for attack generation, judging, quality analysis, and remediation; `claude-haiku-4-5-20251001` for the target runner
- Streamlit

## Project structure

```
prompt-stress-tester/
├── src/
│   ├── attacker.py    # adversarial test case generation
│   ├── runner.py      # test execution against target prompt
│   ├── judge.py       # response scoring and verdict classification
│   ├── analyzer.py    # prompt quality scoring (8-dimension rubric)
│   ├── report.py      # aggregation, gap analysis, remediation
│   └── app.py         # Streamlit UI
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
