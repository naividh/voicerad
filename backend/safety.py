"""
VoiceRad Clinical Safety Rails
Confidence thresholds, referral triggers, human-in-loop flagging,
and CRITICAL/URGENT/ROUTINE triage classification.

This module provides deterministic safety overrides on top of
MedGemma's probabilistic output — ensuring that dangerous findings
are NEVER silently passed through without explicit clinical review.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Triage Levels ──────────────────────────────────────────

class TriageLevel(str, Enum):
    """Clinical triage classification for AI findings."""
    CRITICAL = "CRITICAL"    # Immediate radiologist/clinician review
    URGENT = "URGENT"        # Review within 1 hour
    ROUTINE = "ROUTINE"      # Standard review queue
    NORMAL = "NORMAL"        # No acute findings


# ── Confidence Thresholds ──────────────────────────────────

# Below LOW_CONFIDENCE the system REFUSES to show interpretation
# and forces referral to a human radiologist.
LOW_CONFIDENCE_THRESHOLD = 0.30

# Below UNCERTAIN the system shows interpretation but with
# prominent "LOW CONFIDENCE" warning banner.
UNCERTAIN_THRESHOLD = 0.60

# Above HIGH_CONFIDENCE the system shows green confidence indicator.
# Still requires clinician sign-off (this is NEVER autonomous).
HIGH_CONFIDENCE_THRESHOLD = 0.85


# ── Critical Finding Patterns ──────────────────────────────

# These regex patterns trigger IMMEDIATE referral regardless of
# confidence score. Based on ACR Critical Results guidelines.

CRITICAL_PATTERNS = [
    # Pneumothorax
    r"\b(tension\s+)?pneumothorax\b",
    # Aortic emergencies
    r"\b(aortic\s+dissection|aortic\s+rupture|aortic\s+aneurysm.{0,20}ruptur)\b",
    # Pulmonary embolism
    r"\b(pulmonary\s+embol|PE\s+positive|saddle\s+embol)\b",
    # Stroke
    r"\b(acute\s+(stroke|infarct|ischem)|hemorrhag.{0,10}(stroke|bleed))\b",
    # Active hemorrhage
    r"\b(active\s+(hemorrhag|bleed)|massive\s+hemoptysis)\b",
    # Foreign body / airway
    r"\b(foreign\s+body.{0,15}airway|complete\s+airway\s+obstruct)\b",
    # Cardiac arrest / tamponade
    r"\b(cardiac\s+tamponade|pericardial\s+effusion.{0,20}(large|massive))\b",
    # Necrotizing / gas gangrene
    r"\b(necrotizing\s+fasciitis|gas\s+gangrene|free\s+(air|gas).{0,15}(abdom|periton))\b",
    # Spinal cord compression
    r"\b(spinal\s+cord\s+compress|cauda\s+equina)\b",
    # Testicular / ovarian torsion
    r"\b(testicular\s+torsion|ovarian\s+torsion)\b",
]

URGENT_PATTERNS = [
    # Fractures with displacement
    r"\b(displac.{0,10}fracture|open\s+fracture|pathologic.{0,10}fracture)\b",
    # Pleural effusion (large)
    r"\b(large|massive)\s+pleural\s+effusion\b",
    # Consolidation with clinical concern
    r"\b(lobar\s+consolidation|multilobar\s+pneumonia)\b",
    # Mass / suspected malignancy
    r"\b(suspect.{0,15}(malignan|neoplas|carcinoma|metasta)|lung\s+mass)\b",
    # Bowel obstruction
    r"\b(bowel\s+obstruct|small\s+bowel\s+obstruct|SBO)\b",
    # DVT
    r"\b(deep\s+vein\s+thrombos|DVT)\b",
]

# Patterns that indicate the model is uncertain / hedging
HEDGING_PATTERNS = [
    r"\b(cannot\s+(determine|exclude|rule\s+out))\b",
    r"\b(uncertain|indeterminate|equivocal|ambiguous)\b",
    r"\b(possible|possibly|may\s+represent|could\s+be)\b",
    r"\b(limited\s+(study|exam|evaluation))\b",
    r"\b(further\s+(evaluation|imaging|workup)\s+(recommend|suggest|need))\b",
    r"\b(clinical\s+correlation\s+(is\s+)?(recommend|essential|needed))\b",
]


# ── Safety Assessment Result ───────────────────────────────

@dataclass
class SafetyAssessment:
    """Complete safety evaluation of an AI interpretation."""

    # Triage
    triage_level: TriageLevel = TriageLevel.ROUTINE
    triage_reason: str = ""

    # Confidence
    confidence_score: float = 0.5
    confidence_label: str = "MODERATE"  # LOW / MODERATE / HIGH

    # Flags
    requires_human_review: bool = True   # ALWAYS true for clinical AI
    is_blocked: bool = False             # True = don't show interpretation
    block_reason: str = ""

    # Referral
    referral_triggered: bool = False
    referral_type: str = ""              # "IMMEDIATE" / "URGENT" / ""
    referral_reason: str = ""

    # Detected critical/urgent patterns
    critical_findings: list = field(default_factory=list)
    urgent_findings: list = field(default_factory=list)
    hedging_indicators: list = field(default_factory=list)

    # Warnings for display
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "triage_level": self.triage_level.value,
            "triage_reason": self.triage_reason,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_label": self.confidence_label,
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


# ── Core Safety Engine ─────────────────────────────────────

class ClinicalSafetyEngine:
    """
    Deterministic safety validator that runs AFTER MedGemma inference.

    Design principle: This module can only ADD warnings and BLOCK output.
    It never removes safety flags or overrides clinical caution.
    Safety is monotonically increasing — more information can only
    make the system MORE cautious, never less.
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

        # Compile regex patterns once
        self._critical_re = [
            re.compile(p, re.IGNORECASE) for p in CRITICAL_PATTERNS
        ]
        self._urgent_re = [
            re.compile(p, re.IGNORECASE) for p in URGENT_PATTERNS
        ]
        self._hedging_re = [
            re.compile(p, re.IGNORECASE) for p in HEDGING_PATTERNS
        ]

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
            confidence_score: Model confidence (0-1), or None to estimate
            imaging_type: Type of medical image (CXR, CT, etc.)

        Returns:
            SafetyAssessment with all flags, warnings, and triage level
        """
        result = SafetyAssessment()
        result.requires_human_review = True  # ALWAYS

        # Step 1: Estimate confidence if not provided
        if confidence_score is not None:
            result.confidence_score = max(0.0, min(1.0, confidence_score))
        else:
            result.confidence_score = self._estimate_confidence(interpretation)

        # Step 2: Set confidence label
        result.confidence_label = self._label_confidence(result.confidence_score)

        # Step 3: Scan for critical findings
        result.critical_findings = self._scan_patterns(
            interpretation, self._critical_re
        )
        result.urgent_findings = self._scan_patterns(
            interpretation, self._urgent_re
        )
        result.hedging_indicators = self._scan_patterns(
            interpretation, self._hedging_re
        )

        # Step 4: Determine triage level
        if result.critical_findings:
            result.triage_level = TriageLevel.CRITICAL
            result.triage_reason = (
                f"Critical finding detected: {result.critical_findings[0]}"
            )
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
            result.triage_reason = (
                f"Urgent finding detected: {result.urgent_findings[0]}"
            )
            result.referral_triggered = True
            result.referral_type = "URGENT"
            result.referral_reason = (
                "Urgent finding requires radiologist review within 1 hour."
            )
            result.warnings.append(
                "URGENT finding - Radiologist review within 1 hour"
            )
        elif result.confidence_score < self.uncertain:
            result.triage_level = TriageLevel.ROUTINE
            result.triage_reason = "Low confidence interpretation"
        else:
            # Check for truly negative findings
            neg_patterns = [
                r"no\s+acute", r"normal", r"unremarkable",
                r"no\s+(significant|abnormal)",
            ]
            is_normal = any(
                re.search(p, interpretation, re.IGNORECASE)
                for p in neg_patterns
            )
            result.triage_level = (
                TriageLevel.NORMAL if is_normal else TriageLevel.ROUTINE
            )
            result.triage_reason = (
                "No acute findings" if is_normal
                else "Findings require standard review"
            )

        # Step 5: Apply confidence-based blocking
        if result.confidence_score < self.low_confidence:
            result.is_blocked = True
            result.block_reason = (
                f"Confidence too low ({result.confidence_score:.0%}). "
                "AI interpretation withheld. "
                "Please refer to a radiologist for this image."
            )
            result.referral_triggered = True
            result.referral_type = result.referral_type or "URGENT"
            result.referral_reason = result.referral_reason or (
                "AI confidence below minimum threshold. "
                "Human interpretation required."
            )
            result.warnings.append(
                "AI confidence below safety threshold - interpretation blocked"
            )

        # Step 6: Add hedging warnings
        if result.hedging_indicators:
            result.warnings.append(
                f"Model expressed uncertainty: "
                f"{', '.join(result.hedging_indicators[:3])}"
            )
            # Reduce confidence if model is hedging
            hedging_penalty = len(result.hedging_indicators) * 0.05
            result.confidence_score = max(
                0.1, result.confidence_score - hedging_penalty
            )
            result.confidence_label = self._label_confidence(
                result.confidence_score
            )

        # Step 7: Add mandatory disclaimers
        result.warnings.append(
            "AI-assisted interpretation only. "
            "Clinician review and sign-off required."
        )

        logger.info(
            "Safety assessment: triage=%s confidence=%.2f blocked=%s "
            "critical=%d urgent=%d hedging=%d",
            result.triage_level.value,
            result.confidence_score,
            result.is_blocked,
            len(result.critical_findings),
            len(result.urgent_findings),
            len(result.hedging_indicators),
        )

        return result

    # ── Internal Methods ────────────────────────────────────

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence from interpretation text when model
        doesn't provide an explicit score.

        Heuristic based on:
        - Length of response (very short = uncertain)
        - Hedging language count
        - Specificity of findings
        - Structure quality
        """
        score = 0.65  # Baseline

        # Penalty for very short responses
        word_count = len(text.split())
        if word_count < 20:
            score -= 0.20
        elif word_count < 50:
            score -= 0.10

        # Penalty for hedging
        hedging_count = len(
            self._scan_patterns(text, self._hedging_re)
        )
        score -= hedging_count * 0.07

        # Bonus for structured output
        structure_markers = [
            "FINDINGS", "IMPRESSION", "TECHNIQUE",
            "RECOMMENDATION", "CONCLUSION",
        ]
        structure_count = sum(
            1 for m in structure_markers
            if m in text.upper()
        )
        score += structure_count * 0.03

        # Bonus for specific anatomical references
        anatomy_terms = [
            r"\b(lobe|hilum|hiliar|mediastin|costophrenic|cardiomeg)\b",
            r"\b(opacity|effusion|consolidat|atelectas|infiltrat)\b",
            r"\b(fracture|disloc|sublux|alignment)\b",
        ]
        anatomy_count = sum(
            1 for p in anatomy_terms
            if re.search(p, text, re.IGNORECASE)
        )
        score += anatomy_count * 0.04

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
        """Scan text for regex patterns, return matched strings."""
        found = []
        for regex in patterns:
            matches = regex.findall(text)
            for m in matches:
                # findall returns tuples for groups
                match_str = m if isinstance(m, str) else m[0]
                if match_str and match_str not in found:
                    found.append(match_str.strip())
        return found


# ── Module-level singleton ─────────────────────────────────

_engine: Optional[ClinicalSafetyEngine] = None


def get_safety_engine() -> ClinicalSafetyEngine:
    """Get or create the singleton safety engine."""
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
    return get_safety_engine().assess(
        interpretation, confidence_score, imaging_type
    )
