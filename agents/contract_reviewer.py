"""
Contract Reviewer Agent
========================
Analyzes a contract document and returns:
  - A risk rating (LOW / MEDIUM / HIGH)
  - A list of flagged clauses with explanations
  - A plain-English summary of key obligations
"""
from dataclasses import dataclass, field
from typing import List
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm

DISCLAIMER = (
    "⚠️ This analysis is for informational purposes only and does not constitute legal advice. "
    "Please consult a qualified attorney for advice specific to your situation."
)

_REVIEW_PROMPT = PromptTemplate(
    input_variables=["contract_text"],
    template="""You are an expert contract review attorney. Analyze the following contract carefully.

CONTRACT:
{contract_text}

Provide a structured analysis in EXACTLY this format:

RISK_LEVEL: [LOW | MEDIUM | HIGH]

FLAGGED_CLAUSES:
- Clause: [quote the problematic clause]
  Risk: [explain why it is risky]
  Recommendation: [what to do]

KEY_OBLIGATIONS:
- [list the main obligations of each party in plain English]

MISSING_STANDARD_CLAUSES:
- [list any important clauses that appear to be absent, e.g., limitation of liability, dispute resolution]

SUMMARY:
[2-3 sentence plain-English summary of the overall contract and its main risk profile]

Analysis:""",
)


@dataclass
class ContractReviewResult:
    risk_level: str = "UNKNOWN"
    flagged_clauses: List[dict] = field(default_factory=list)
    key_obligations: List[str] = field(default_factory=list)
    missing_clauses: List[str] = field(default_factory=list)
    summary: str = ""
    raw_analysis: str = ""
    disclaimer: str = DISCLAIMER


def _parse_review(raw: str) -> ContractReviewResult:
    """Parse the structured LLM output into a ContractReviewResult."""
    result = ContractReviewResult(raw_analysis=raw)
    lines = raw.splitlines()
    section = None
    current_clause: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("RISK_LEVEL:"):
            level = stripped.split(":", 1)[1].strip().upper()
            result.risk_level = level if level in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM"

        elif stripped == "FLAGGED_CLAUSES:":
            section = "clauses"

        elif stripped == "KEY_OBLIGATIONS:":
            if current_clause:
                result.flagged_clauses.append(current_clause)
                current_clause = None
            section = "obligations"

        elif stripped == "MISSING_STANDARD_CLAUSES:":
            section = "missing"

        elif stripped == "SUMMARY:":
            section = "summary"

        elif section == "clauses":
            if stripped.startswith("- Clause:"):
                if current_clause:
                    result.flagged_clauses.append(current_clause)
                current_clause = {"clause": stripped[len("- Clause:"):].strip(), "risk": "", "recommendation": ""}
            elif stripped.startswith("Risk:") and current_clause:
                current_clause["risk"] = stripped[len("Risk:"):].strip()
            elif stripped.startswith("Recommendation:") and current_clause:
                current_clause["recommendation"] = stripped[len("Recommendation:"):].strip()

        elif section == "obligations":
            if stripped.startswith("- "):
                result.key_obligations.append(stripped[2:])

        elif section == "missing":
            if stripped.startswith("- "):
                result.missing_clauses.append(stripped[2:])

        elif section == "summary":
            result.summary += (" " + stripped) if result.summary else stripped

    if current_clause:
        result.flagged_clauses.append(current_clause)

    return result


def review_contract(contract_text: str) -> ContractReviewResult:
    """Main entry point: analyze a contract text and return structured findings."""
    llm = get_primary_llm()
    prompt = _REVIEW_PROMPT.format(contract_text=contract_text[:12000])  # stay within context
    raw = llm.invoke(prompt)
    return _parse_review(raw)
