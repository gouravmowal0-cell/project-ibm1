"""
Compliance Checker Agent
=========================
Checks a document against a specified regulatory framework and reports:
  - Overall compliance status
  - Per-requirement compliance findings
  - Remediation recommendations
"""
from dataclasses import dataclass, field
from typing import List
from langchain_core.prompts import PromptTemplate
from core.llm import get_primary_llm

DISCLAIMER = (
    "⚠️ This compliance check is for informational purposes only. "
    "Regulatory requirements change frequently — always verify with official sources or legal counsel."
)

SUPPORTED_FRAMEWORKS = [
    "GDPR",       # General Data Protection Regulation
    "HIPAA",      # Health Insurance Portability and Accountability Act
    "CCPA",       # California Consumer Privacy Act
    "SOX",        # Sarbanes-Oxley Act
    "PCI-DSS",    # Payment Card Industry Data Security Standard
    "ISO 27001",  # Information security management
    "GENERAL",    # Generic compliance check
]

_COMPLIANCE_PROMPT = PromptTemplate(
    input_variables=["framework", "document_text"],
    template="""You are a regulatory compliance expert specializing in {framework}.

Analyze the following document for compliance with {framework} requirements.

DOCUMENT:
{document_text}

Provide your analysis in EXACTLY this format:

COMPLIANCE_STATUS: [COMPLIANT | PARTIALLY_COMPLIANT | NON_COMPLIANT]

COMPLIANCE_SCORE: [0-100]

FINDINGS:
- Requirement: [specific {framework} requirement or article]
  Status: [PASS | FAIL | PARTIAL | N/A]
  Finding: [what was found in the document]
  Recommendation: [action to achieve compliance]

CRITICAL_GAPS:
- [list any critical compliance gaps that pose immediate risk]

POSITIVE_FINDINGS:
- [list areas where the document demonstrates good compliance]

SUMMARY:
[Plain-English summary of compliance posture and priority actions]

Analysis:""",
)


@dataclass
class ComplianceFinding:
    requirement: str = ""
    status: str = ""
    finding: str = ""
    recommendation: str = ""


@dataclass
class ComplianceResult:
    framework: str = ""
    compliance_status: str = "UNKNOWN"
    compliance_score: int = 0
    findings: List[ComplianceFinding] = field(default_factory=list)
    critical_gaps: List[str] = field(default_factory=list)
    positive_findings: List[str] = field(default_factory=list)
    summary: str = ""
    raw_analysis: str = ""
    disclaimer: str = DISCLAIMER


def _parse_compliance(raw: str, framework: str) -> ComplianceResult:
    result = ComplianceResult(framework=framework, raw_analysis=raw)
    lines = raw.splitlines()
    section = None
    current_finding: ComplianceFinding | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("COMPLIANCE_STATUS:"):
            val = stripped.split(":", 1)[1].strip().upper()
            if val in {"COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT"}:
                result.compliance_status = val

        elif stripped.startswith("COMPLIANCE_SCORE:"):
            try:
                result.compliance_score = int(stripped.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

        elif stripped == "FINDINGS:":
            section = "findings"

        elif stripped == "CRITICAL_GAPS:":
            if current_finding:
                result.findings.append(current_finding)
                current_finding = None
            section = "critical"

        elif stripped == "POSITIVE_FINDINGS:":
            section = "positive"

        elif stripped == "SUMMARY:":
            section = "summary"

        elif section == "findings":
            if stripped.startswith("- Requirement:"):
                if current_finding:
                    result.findings.append(current_finding)
                current_finding = ComplianceFinding(
                    requirement=stripped[len("- Requirement:"):].strip()
                )
            elif stripped.startswith("Status:") and current_finding:
                current_finding.status = stripped[len("Status:"):].strip()
            elif stripped.startswith("Finding:") and current_finding:
                current_finding.finding = stripped[len("Finding:"):].strip()
            elif stripped.startswith("Recommendation:") and current_finding:
                current_finding.recommendation = stripped[len("Recommendation:"):].strip()

        elif section == "critical":
            if stripped.startswith("- "):
                result.critical_gaps.append(stripped[2:])

        elif section == "positive":
            if stripped.startswith("- "):
                result.positive_findings.append(stripped[2:])

        elif section == "summary":
            result.summary += (" " + stripped) if result.summary else stripped

    if current_finding:
        result.findings.append(current_finding)

    return result


def check_compliance(document_text: str, framework: str = "GENERAL") -> ComplianceResult:
    """Check a document against the given regulatory framework."""
    framework = framework.upper()
    llm = get_primary_llm()
    prompt = _COMPLIANCE_PROMPT.format(
        framework=framework,
        document_text=document_text[:10000],
    )
    raw = llm.invoke(prompt)
    return _parse_compliance(raw, framework)
