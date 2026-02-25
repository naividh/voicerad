"""
Tests for VoiceRad Clinical Safety Rails

Run: python -m pytest tests/test_safety.py -v
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from safety import (
    ClinicalSafetyEngine,
    SafetyAssessment,
    TriageLevel,
    assess_interpretation,
)


class TestTriageLevels:
    """Test that critical/urgent findings are correctly detected."""

    def setup_method(self):
        self.engine = ClinicalSafetyEngine()

    def test_tension_pneumothorax_is_critical(self):
        text = "Large left-sided tension pneumothorax with mediastinal shift."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.CRITICAL
        assert result.referral_triggered is True
        assert result.referral_type == "IMMEDIATE"
        assert len(result.critical_findings) > 0

    def test_aortic_dissection_is_critical(self):
        text = "Findings consistent with acute aortic dissection type A."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.CRITICAL
        assert result.referral_triggered is True

    def test_pulmonary_embolism_is_critical(self):
        text = "Saddle embolus identified in the main pulmonary artery."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.CRITICAL

    def test_cardiac_tamponade_is_critical(self):
        text = "Large pericardial effusion with signs of cardiac tamponade."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.CRITICAL

    def test_lobar_pneumonia_is_urgent(self):
        text = "Right lower lobar consolidation consistent with lobar pneumonia."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.URGENT
        assert result.referral_triggered is True
        assert result.referral_type == "URGENT"

    def test_suspected_malignancy_is_urgent(self):
        text = "2.3 cm spiculated lung mass in right upper lobe, suspect malignancy."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.URGENT

    def test_large_pleural_effusion_is_urgent(self):
        text = "Large left-sided pleural effusion with compressive atelectasis."
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.URGENT

    def test_normal_finding_is_normal(self):
        text = (
            "FINDINGS: Heart size normal. Lungs clear bilaterally. "
            "No pleural effusion. No acute abnormality identified. "
            "IMPRESSION: Normal chest radiograph."
        )
        result = self.engine.assess(text)
        assert result.triage_level == TriageLevel.NORMAL

    def test_mild_finding_is_routine(self):
        text = (
            "FINDINGS: Mild degenerative changes of the thoracic spine. "
            "Heart size upper limits of normal. "
            "IMPRESSION: Age-appropriate findings."
        )
        result = self.engine.assess(text)
        assert result.triage_level in (TriageLevel.ROUTINE, TriageLevel.NORMAL)


class TestConfidenceThresholds:
    """Test confidence scoring and blocking behavior."""

    def setup_method(self):
        self.engine = ClinicalSafetyEngine()

    def test_very_low_confidence_blocks(self):
        result = self.engine.assess("Maybe something.", confidence_score=0.15)
        assert result.is_blocked is True
        assert result.confidence_label == "VERY_LOW"
        assert len(result.block_reason) > 0

    def test_low_confidence_warns(self):
        result = self.engine.assess(
            "Possible opacity in left lower lobe.",
            confidence_score=0.45,
        )
        assert result.is_blocked is False
        assert result.confidence_label == "LOW"

    def test_moderate_confidence_passes(self):
        result = self.engine.assess(
            "Right lower lobe consolidation with air bronchograms. "
            "IMPRESSION: Pneumonia.",
            confidence_score=0.70,
        )
        assert result.is_blocked is False
        assert result.confidence_label == "MODERATE"

    def test_high_confidence_green(self):
        result = self.engine.assess(
            "Clear lungs. Normal heart size. No acute findings. "
            "IMPRESSION: Normal chest radiograph.",
            confidence_score=0.92,
        )
        assert result.confidence_label == "HIGH"
        assert result.is_blocked is False

    def test_confidence_always_requires_review(self):
        """Even high confidence must require human review."""
        result = self.engine.assess("Normal.", confidence_score=0.99)
        assert result.requires_human_review is True

    def test_confidence_estimation_short_text(self):
        """Short, vague text should have low estimated confidence."""
        result = self.engine.assess("Maybe normal.")
        assert result.confidence_score < 0.6

    def test_confidence_estimation_detailed_text(self):
        """Detailed structured text should have higher confidence."""
        result = self.engine.assess(
            "TECHNIQUE: PA and lateral chest radiograph. "
            "FINDINGS: Heart size is normal. Mediastinal contours are "
            "unremarkable. The lungs are clear bilaterally without focal "
            "consolidation, pleural effusion, or pneumothorax. "
            "Costophrenic angles are sharp. Osseous structures intact. "
            "IMPRESSION: No acute cardiopulmonary abnormality."
        )
        assert result.confidence_score > 0.6


class TestHedgingDetection:
    """Test that model uncertainty language is caught."""

    def setup_method(self):
        self.engine = ClinicalSafetyEngine()

    def test_hedging_detected(self):
        text = (
            "Cannot exclude a small pneumothorax. "
            "Further evaluation with CT recommended. "
            "Clinical correlation is recommended."
        )
        result = self.engine.assess(text)
        assert len(result.hedging_indicators) >= 2
        assert any("cannot" in h.lower() for h in result.hedging_indicators)

    def test_hedging_reduces_confidence(self):
        """Hedging language should reduce the confidence score."""
        text_certain = "Right lower lobe consolidation. Pneumonia."
        text_hedging = (
            "Possibly right lower lobe consolidation. "
            "Cannot rule out pneumonia. "
            "Clinical correlation recommended."
        )
        r_certain = self.engine.assess(text_certain, confidence_score=0.80)
        r_hedging = self.engine.assess(text_hedging, confidence_score=0.80)
        assert r_hedging.confidence_score < r_certain.confidence_score


class TestSafetyMonotonicity:
    """Test that safety can only increase, never decrease."""

    def setup_method(self):
        self.engine = ClinicalSafetyEngine()

    def test_human_review_always_required(self):
        """requires_human_review must always be True."""
        texts = [
            "Normal chest radiograph.",
            "Tension pneumothorax.",
            "Maybe something.",
            "",
        ]
        for text in texts:
            result = self.engine.assess(text)
            assert result.requires_human_review is True


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def setup_method(self):
        self.engine = ClinicalSafetyEngine()

    def test_empty_text(self):
        result = self.engine.assess("")
        assert result.requires_human_review is True
        assert result.confidence_score < 0.5

    def test_non_medical_text(self):
        result = self.engine.assess("The weather is nice today.")
        assert result.requires_human_review is True

    def test_very_long_text(self):
        text = "Normal findings. " * 500
        result = self.engine.assess(text)
        assert result.requires_human_review is True

    def test_convenience_function(self):
        """Test the module-level assess_interpretation function."""
        result = assess_interpretation(
            "Tension pneumothorax.", confidence_score=0.90
        )
        assert result.triage_level == TriageLevel.CRITICAL
        assert result.referral_triggered is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = self.engine.assess("Normal.")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "triage_level" in d
        assert "confidence_score" in d
        assert "warnings" in d
        assert isinstance(d["warnings"], list)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
