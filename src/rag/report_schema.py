"""
src/rag/report_schema.py
---------------------------
The structured output contract for a generated threat report. The LLM is
instructed (see prompt_templates.py) to return JSON matching this shape
directly -- we parse and validate against it rather than regex-scraping
free-text sections, so a malformed/incomplete LLM response fails loudly
(ReportParseError) instead of silently producing a half-populated report.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ReportParseError(Exception):
    """Raised when an LLM response cannot be parsed into a ThreatReport.
    Callers should treat this as a failed generation, not retry-and-hope
    silently, and should surface the raw response for debugging."""
    pass


@dataclass
class TechniqueFinding:
    technique_id: str
    technique_name: str
    tactic: str
    similarity_score: float          # carried through unchanged from Stage 7 -- the LLM does not invent this number
    confidence_label: str            # "CANDIDATE" | "NO_CONFIDENT_MATCH" -- carried through unchanged
    reasoning: str                   # LLM's plain-language justification, grounded in the context it was given

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ThreatReport:
    cluster_id: int
    title: str
    summary: str
    observed_behavior: str
    likely_techniques: List[TechniqueFinding]
    risk_assessment: str

    # provenance -- not shown to the end user necessarily, but kept so
    # every report is traceable back to the exact inputs that produced it
    generated_by: str = "unknown"                  # LLM backend identifier, e.g. "ollama:llama3" or "template"
    grounding_event_uids: List[str] = field(default_factory=list)
    cluster_size: Optional[int] = None
    max_novelty_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["likely_techniques"] = [t.as_dict() for t in self.likely_techniques]
        return d


REQUIRED_LLM_JSON_FIELDS = ["title", "summary", "observed_behavior", "likely_techniques", "risk_assessment"]


def parse_llm_json_response(raw_json: Dict[str, Any], cluster_id: int, generated_by: str,
                             grounding_event_uids: List[str], cluster_size: int,
                             max_novelty_score: Optional[float],
                             allowed_technique_ids: Optional[List[str]] = None) -> ThreatReport:
    """
    Validates the LLM's JSON output against REQUIRED_LLM_JSON_FIELDS and,
    critically, checks every cited technique_id against
    allowed_technique_ids (the exact set of technique_ids that were
    actually present in the context the LLM was given -- see
    context_builder.py). A technique_id NOT in that set means the model
    cited something it was never shown, i.e. a hallucination -- this
    raises ReportParseError rather than silently keeping it.
    """
    missing = [f for f in REQUIRED_LLM_JSON_FIELDS if f not in raw_json]
    if missing:
        raise ReportParseError(f"LLM response missing required field(s): {missing}. Raw: {raw_json}")

    techniques = []
    for t in raw_json["likely_techniques"]:
        for req in ("technique_id", "technique_name", "tactic", "similarity_score", "confidence_label", "reasoning"):
            if req not in t:
                raise ReportParseError(f"Technique entry missing '{req}': {t}")
        if allowed_technique_ids is not None and t["technique_id"] not in allowed_technique_ids:
            raise ReportParseError(
                f"LLM cited technique_id '{t['technique_id']}' which was NOT present in the retrieved "
                f"context (allowed: {allowed_technique_ids}) -- this is a hallucinated technique and "
                f"the report is rejected rather than returned."
            )
        techniques.append(TechniqueFinding(
            technique_id=t["technique_id"], technique_name=t["technique_name"], tactic=t["tactic"],
            similarity_score=float(t["similarity_score"]), confidence_label=t["confidence_label"],
            reasoning=t["reasoning"],
        ))

    return ThreatReport(
        cluster_id=cluster_id,
        title=raw_json["title"],
        summary=raw_json["summary"],
        observed_behavior=raw_json["observed_behavior"],
        likely_techniques=techniques,
        risk_assessment=raw_json["risk_assessment"],
        generated_by=generated_by,
        grounding_event_uids=grounding_event_uids,
        cluster_size=cluster_size,
        max_novelty_score=max_novelty_score,
    )
