use crate::models::{EvidenceKind, IntentNode};

#[cfg(test)]
use crate::models::Evidence;

/// Weights for stability computation.
/// These are tunable — start conservative and adjust based on real usage.
pub struct StabilityWeights {
    pub base: f64,
    pub test_pass: f64,
    pub test_pass_cap: f64,
    pub code_committed: f64,
    pub consumed_by_other: f64,
    pub consumed_cap: f64,
    pub conflict_penalty: f64,
    pub manual_approval: f64,
}

impl Default for StabilityWeights {
    fn default() -> Self {
        Self {
            base: 0.3,
            test_pass: 0.05,
            test_pass_cap: 0.3,
            code_committed: 0.2,
            consumed_by_other: 0.1,
            consumed_cap: 0.2,
            conflict_penalty: 0.15,
            manual_approval: 0.3,
        }
    }
}

pub struct StabilityScorer {
    weights: StabilityWeights,
}

impl StabilityScorer {
    pub fn new() -> Self {
        Self {
            weights: StabilityWeights::default(),
        }
    }

    pub fn with_weights(weights: StabilityWeights) -> Self {
        Self { weights }
    }

    /// Compute stability score for an intent based on its evidence.
    /// Returns a value in [0.0, 1.0].
    pub fn compute(&self, intent: &IntentNode) -> f64 {
        let mut score = self.weights.base;

        let w = &self.weights;

        // Tests passing increases confidence
        let test_passes = intent
            .evidence
            .iter()
            .filter(|e| e.kind == EvidenceKind::TestPass)
            .count() as f64;
        score += (test_passes * w.test_pass).min(w.test_pass_cap);

        // Code committed (not just planned) increases confidence
        let has_committed = intent
            .evidence
            .iter()
            .any(|e| e.kind == EvidenceKind::CodeCommitted);
        if has_committed {
            score += w.code_committed;
        }

        // Other agents depending on this increases confidence (network effect)
        let dependents = intent
            .evidence
            .iter()
            .filter(|e| e.kind == EvidenceKind::ConsumedByOther)
            .count() as f64;
        score += (dependents * w.consumed_by_other).min(w.consumed_cap);

        // Conflicts decrease confidence
        let conflicts = intent
            .evidence
            .iter()
            .filter(|e| e.kind == EvidenceKind::Conflict)
            .count() as f64;
        score -= conflicts * w.conflict_penalty;

        // Manual approval is a strong signal
        let has_approval = intent
            .evidence
            .iter()
            .any(|e| e.kind == EvidenceKind::ManualApproval);
        if has_approval {
            score += w.manual_approval;
        }

        // Test failures are a strong negative signal
        let test_fails = intent
            .evidence
            .iter()
            .filter(|e| e.kind == EvidenceKind::TestFail)
            .count() as f64;
        score -= test_fails * w.conflict_penalty;

        score.clamp(0.0, 1.0)
    }

    /// Batch compute stability for multiple intents.
    pub fn compute_batch(&self, intents: &[IntentNode]) -> Vec<(String, f64)> {
        intents
            .iter()
            .map(|intent| (intent.id.clone(), self.compute(intent)))
            .collect()
    }
}

impl Default for StabilityScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_intent(evidence: Vec<Evidence>) -> IntentNode {
        IntentNode {
            evidence,
            ..IntentNode::new("test-agent", "test intent")
        }
    }

    #[test]
    fn test_base_stability() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![]);
        let score = scorer.compute(&intent);
        assert!((score - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_committed_code_increases_stability() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![Evidence::code_committed("initial commit")]);
        let score = scorer.compute(&intent);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tests_passing_increases_stability() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![
            Evidence::test_pass("unit tests"),
            Evidence::test_pass("integration tests"),
            Evidence::test_pass("e2e tests"),
        ]);
        let score = scorer.compute(&intent);
        // base 0.3 + 3 * 0.05 = 0.45
        assert!((score - 0.45).abs() < f64::EPSILON);
    }

    #[test]
    fn test_test_pass_cap() {
        let scorer = StabilityScorer::new();
        // 10 test passes should cap at 0.3 bonus
        let evidence: Vec<Evidence> = (0..10)
            .map(|i| Evidence::test_pass(&format!("test {}", i)))
            .collect();
        let intent = make_intent(evidence);
        let score = scorer.compute(&intent);
        // base 0.3 + cap 0.3 = 0.6
        assert!((score - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conflicts_decrease_stability() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![
            Evidence::code_committed("commit"),
            Evidence::conflict("schema mismatch"),
        ]);
        let score = scorer.compute(&intent);
        // base 0.3 + committed 0.2 - conflict 0.15 = 0.35
        assert!((score - 0.35).abs() < f64::EPSILON);
    }

    #[test]
    fn test_consumed_by_others_increases_stability() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![
            Evidence::code_committed("commit"),
            Evidence::consumed_by("agent-b"),
            Evidence::consumed_by("agent-c"),
        ]);
        let score = scorer.compute(&intent);
        // base 0.3 + committed 0.2 + 2 * 0.1 = 0.7
        assert!((score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_high_stability_scenario() {
        let scorer = StabilityScorer::new();
        let intent = make_intent(vec![
            Evidence::code_committed("commit"),
            Evidence::test_pass("unit tests"),
            Evidence::test_pass("integration tests"),
            Evidence::consumed_by("agent-b"),
            Evidence::consumed_by("agent-c"),
        ]);
        let score = scorer.compute(&intent);
        // base 0.3 + committed 0.2 + 2*0.05 + 2*0.1 = 0.8
        assert!((score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stability_clamped_to_unit_range() {
        let scorer = StabilityScorer::new();
        // Lots of positive evidence
        let mut evidence = vec![Evidence::code_committed("commit")];
        for i in 0..20 {
            evidence.push(Evidence::test_pass(&format!("test {}", i)));
            evidence.push(Evidence::consumed_by(&format!("agent-{}", i)));
        }
        evidence.push(Evidence::manual_approval());
        let intent = make_intent(evidence);
        let score = scorer.compute(&intent);
        assert!(score <= 1.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_stability_floor_at_zero() {
        let scorer = StabilityScorer::new();
        let evidence: Vec<Evidence> = (0..10)
            .map(|i| Evidence::conflict(&format!("conflict {}", i)))
            .collect();
        let intent = make_intent(evidence);
        let score = scorer.compute(&intent);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }
}
