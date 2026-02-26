"""
VoiceRad Clinical Safety Rails (v1.3.0)
Confidence thresholds, referral triggers, human-in-loop flagging,
and CRITICAL/URGENT/ROUTINE triage classification.

This module provides deterministic safety overrides on top of
MedGemma's probabilistic output -- ensuring that dangerous findings
are NEVER silently passed through without explicit clinical review.

v1.3.0 changes:
- Added support for model-calibrated confidence (log-prob based)
- Fixed hedging penalty: cautious language is now REWARDED not penalized
- Improved confidence estimation with anatomical specificity scoring
- Added confidence source tracking (model_logprob vs text_heuristic)
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# -- Triage Levels -----------------------------------------------
class TriageLevel(str, Enum):
    """Clinical triage classification for AI findings."""
    CRITICAL = "CRITICAL"   # Immediate radiologist/clinician review
    URGENT   = "URGENT"     # Review within 1 hour
    ROUTINE  = "ROUTINE"    # Standard review queue
    NORMAL   = "NORMAL"     # No acute findings


# -- Confidence Thresholds ----------------------------------------
LOW_CONFIDENCE_THRESHOLD  = 0.30
UNCERTAIN_THRESHOLD       = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.85


# -- Critical Finding Patterns ------------------------------------
CRITICAL_PATTERNS = [
    r"\b(tension\s+)?pneumothorax\b",
    r"\b(aortic\s+dissection|aortic\s+rupture|aortic\s+aneurysm.{0,20}ruptur)\b",
    r"\b(pulmonary\s+embol|PE\s+positive|saddle\s+embol)\b",
    r"\b(acute\s+(stroke|infarct|ischem)|hemorrhag.{0,10}(stroke|bleed))\b",
    r"\b(active\s+(hemorrhag|bleed)|massive\s+hemoptysis)\b",
    r"\b(foreign\s+body.{0,15}airway|complete\s+airway\s+obstruct)\b",
    r"\b(cardiac\s+tamponade|pericardial\s+effusion.{0,20}(large|massive))\b",
    r"\b(necrotizing\s+fasciitis|gas\s+gangrene|free\s+(air|gas).{0,15}(abdom|periton))\b",
    r"\b(spinal\s+cord\s+compress|cauda\s+equina)\b",
    r"\b(testicular\s+torsion|ovarian\s+torsion)\b",
]

URGENT_PATTERNS = [
    r"\b(displac.{0,10}fracture|open\s+fracture|pathologic.{0,10}fracture)\b",
    r"\b(large|massive)\s+pleural\s+effusion\b",
    r"\b(lobar\s+consolidation|multilobar\s+pneumonia)\b",
    r"\b(suspect.{0,15}(malignan|neoplas|carcinoma|metasta)|lung\s+mass)\b",
    r"\b(bowel\s+obstruct|small\s+bowel\s+obstruct|SBO)\b",
    r"\b(deep\s+vein\s+thrombos|DVT)\b",
]

# Patterns indicating model is hedging (this is APPROPRIATE caution)
HEDGING_PATTERNS = [
    r"\b(cannot\s+(determine|exclude|rule\s+out))\b",
    r"\b(uncertain|indeterminate|equivocal|ambiguous)\b",
    r"\b(possible|possibly|may\s+represent|could\s+be)\b",
    r"\b(limited\s+(study|exam|evaluation))\b",
    r"\b(further\s+(evaluation|imaging|workup)\s+(recommend|suggest|need))\b",
    r"\b(clinical\s+correlation\s+(is\s+)?(recommend|essential|needed))\b",
]


# -- Safety Assessment Result -------------------------------------
@dataclass
class SafetyAssessment:
    """Complete safety evaluation of an AI interpretation."""

    triage_level: TriageLevel = TriageLevel.ROUTINE
    triage_reason: str = ""

    confidence_score: float = 0.5
    confidence_label: str = "MODERATE"
    confidence_source: str = "text_heuristic"  # "model_logprob" or "text_heuristic"

    requires_human_review: bool = True
    is_blocked: bool = False
    block_reason: str = ""

    referral_triggered: bool = False
    referral_type: str = ""
    referral_reason: str = ""

    critical_findings: list = field(default_factory=list)
    urgent_findings: list = field(default_factory=list)
    hedging_indicators: list = field(default_factory=list)

    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "triage_level": self.triage_level.value,
            "triage_reason": self.triage_reason,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_label": self.confidence_label,
            "confidence_source": self.confidence_source,
            "requires_human_review": self.requires_human_review,
            "is_blocked": self.is_blocked,
            "block_reason": self.block_reason,
            "referral_triggered": self.referral_triggered,
            "referral_type": self.referral_type,
            "referral_reason": self.referral_reason,
            "critical_findings": self.critical_findings,
            "urgent_findings": self.urgent_findings,
            "hedging_indicators": self.hedging_indicators,
            "warnings": self.warnings,
        }


# -- Core Safety Engine -------------------------------------------
class ClinicalSafetyEngine:
    """
    Deterministic safety validator that runs AFTER MedGemma inference.

    Design principles:
    - This module can only ADD warnings and BLOCK output.
    - It never removes safety flags or overrides clinical caution.
    - Safety is monotonically increasing.
    - Appropriately cautious language (hedging) is treated as a sign
      of good calibration, NOT penalized.
    """

    def __init__(
        self,
        low_confidence: float = LOW_CONFIDENCE_THRESHOLD,
        uncertain: float = UNCERTAIN_THRESHOLD,
        high_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
    ):
        self.low_confidence = low_confidence
        self.uncertain = uncertain
        self.high_confidence = high_confidence

        self._critical_re = [re.compile(p, re.IGNORECASE) for p in CRITICAL_PATTERNS]
        self._urgent_re   = [re.compile(p, re.IGNORECASE) for p in URGENT_PATTERNS]
        self._hedging_re  = [re.compile(p, re.IGNORECASE) for p in HEDGING_PATTERNS]

    def assess(
        self,
        interpretation: str,
        confidence_score: Optional[float] = None,
        imaging_type: str = "Unknown",
    ) -> SafetyAssessment:
        """
        Run full safety assessment on an AI interpretation.

        Args:
            interpretation: Raw text output from MedGemma
            confidence_score: Model confidence (0-1) from log-probs, or None
            imaging_type: Type of medical image
        """
        result = SafetyAssessment()
        result.requires_human_review = True  # ALWAYS

        # Step 1: Set confidence -- prefer model-provided score
        if confidence_score is not None:
            result.confidence_score = max(0.0, min(1.0, confidence_score))
            result.confidence_source = "model_logprob"
        else:
            result.confidence_score = self._estimate_confidence(interpretation)
            result.confidence_source = "text_heuristic"

        result.confidence_label = self._label_confidence(result.confidence_score)

        # Step 2: Scan for critical/urgent findings
        result.critical_findings = self._scan_patterns(interpretation, self._critical_re)
        result.urgent_findings = self._scan_patterns(interpretation, self._urgent_re)
        result.hedging_indicators = self._scan_patterns(interpretation, self._hedging_re)

        # Step 3: Determine triage level
        if result.critical_findings:
            result.triage_level = TriageLevel.CRITICAL
            result.triage_reason = f"Critical finding detected: {result.critical_findings[0]}"
            result.referral_triggered = True
            result.referral_type = "IMMEDIATE"
            result.referral_reason = (
                "CRITICAL finding requires immediate radiologist review. "
                "Do NOT act on AI interpretation alone."
            )
            result.warnings.append(
                "CRITICAL FINDING DETECTED - Immediate radiologist review required"
            )
        elif result.urgent_findings:
            result.triage_level = TriageLevel.URGENT
            result.triage_reason = f"Urgent finding detected: {result.urgent_findings[0]}"
            result.referral_triggered = True
            result.referral_type = "URGENT"
            result.referral_reason = "Urgent finding requires radiologist review within 1 hour."
            result.warnings.append("URGENT finding - Radiologist review within 1 hour")
        elif result.confidence_score < self.uncertain:
            result.triage_level = TriageLevel.ROUTINE
            result.triage_reason = "Low confidence interpretation"
        else:
            neg_patterns = [r"no\s+acute", r"normal", r"unremarkable", r"no\s+(significant|abnormal)"]
            is_normal = any(re.search(p, interpretation, re.IGNORECASE) for p in neg_patterns)
            result.triage_level = TriageLevel.NORMAL if is_normal else TriageLevel.ROUTINE
            result.triage_reason = "No acute findings" if is_normal else "Findings require standard review"

        # Step 4: Confidence-based blocking
        if result.confidence_score < self.low_confidence:
            result.is_blocked = True
            result.block_reason = (
                f"Confidence too low ({result.confidence_score:.0%}). "
                "AI interpretation withheld. Refer to a radiologist."
            )
            result.referral_triggered = True
            result.referral_type = result.referral_type or "URGENT"
            result.referral_reason = result.referral_reason or (
                "AI confidence below minimum threshold. Human interpretation required."
            )
            result.warnings.append("AI confidence below safety threshold - interpretation blocked")

        # Step 5: Hedging indicators -- note but do NOT penalize
        # Appropriate hedging is a sign of good model calibration.
        # We flag it for clinician awareness but do not reduce confidence.
        if result.hedging_indicators:
            result.warnings.append(
                f"Model expressed appropriate caution: {', '.join(result.hedging_indicators[:3])}"
            )
            # Only reduce confidence if using text heuristic AND excessive hedging
            if result.confidence_source == "text_heuristic" and len(result.hedging_indicators) > 4:
                penalty = (len(result.hedging_indicators) - 4) * 0.03
                result.confidence_score = max(0.1, result.confidence_score - penalty)
                result.confidence_label = self._label_confidence(result.confidence_score)

        # Step 6: Mandatory disclaimers
        result.warnings.append(
            "AI-assisted interpretation only. Clinician review and sign-off required."
        )

        logger.info(
            "Safety: triage=%s conf=%.2f(%s) blocked=%s crit=%d urg=%d hedge=%d",
            result.triage_level.value, result.confidence_score,
            result.confidence_source, result.is_blocked,
            len(result.critical_findings), len(result.urgent_findings),
            len(result.hedging_indicators),
        )
        return result

    # -- Internal Methods -----------------------------------------

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence from text when model log-probs unavailable.

        Multi-factor heuristic based on:
        - Response length and detail
        - Anatomical specificity
        - Structure quality (FINDINGS/IMPRESSION sections)
        - Hedging language (moderate hedging is appropriate, not penalized)
        """
        score = 0.55  # Conservative baseline

        word_count = len(text.split())
        if word_count < 15:
            score -= 0.20
        elif word_count < 40:
            score -= 0.10
        elif word_count > 100:
            score += 0.10
        elif word_count > 200:
            score += 0.15

        # Bonus for structured output
        markers = ["FINDINGS", "IMPRESSION", "TECHNIQUE", "RECOMMENDATION", "CONCLUSION"]
        score += sum(0.04 for m in markers if m in text.upper())

        # Bonus for anatomical specificity
        anatomy = [
            r"\b(lobe|hilum|hilar|mediastin|costophrenic|cardiomeg)\b",
            r"\b(opacity|effusion|consolidat|atelectas|infiltrat)\b",
            r"\b(fracture|disloc|sublux|alignment)\b",
            r"\b(right|left|bilateral|upper|lower|middle)\b",
        ]
        score += sum(0.03 for p in anatomy if re.search(p, text, re.IGNORECASE))

        return max(0.05, min(0.95, score))

    def _label_confidence(self, score: float) -> str:
        if score >= self.high_confidence:
            return "HIGH"
        elif score >= self.uncertain:
            return "MODERATE"
        elif score >= self.low_confidence:
            return "LOW"
        else:
            return "VERY_LOW"

    @staticmethod
    def _scan_patterns(text: str, patterns: list) -> list:
        found = []
        for regex in patterns:
            matches = regex.findall(text)
            for m in matches:
                match_str = m if isinstance(m, str) else m[0]
                if match_str and match_str not in found:
                    found.append(match_str.strip())
        return found


# -- Module-level singleton ----------------------------------------
_engine: Optional[ClinicalSafetyEngine] = None


def get_safety_engine() -> ClinicalSafetyEngine:
    global _engine
    if _engine is None:
        _engine = ClinicalSafetyEngine()
    return _engine


def assess_interpretation(
    interpretation: str,
    confidence_score: Optional[float] = None,
    imaging_type: str = "Unknown",
) -> SafetyAssessment:
    """Convenience function for quick safety assessment."""
    return get_safety_engine().assess(interpretation, confidence_score, imaging_type)
