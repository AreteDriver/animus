use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::matching;

/// A single unit of semantic intent in the shared graph.
/// Published by agents as they make architectural decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentNode {
    pub id: String,
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,

    /// Human-readable description of the decision
    pub intent: String,

    /// Interfaces this decision provides to other scopes
    pub provides: Vec<InterfaceSpec>,

    /// Interfaces this decision requires from other scopes
    pub requires: Vec<InterfaceSpec>,

    /// Constraints this decision imposes on other agents
    pub constraints: Vec<Constraint>,

    /// Confidence that this decision is final (0.0 = exploring, 1.0 = committed)
    pub stability: f64,

    /// Evidence supporting the stability score
    pub evidence: Vec<Evidence>,

    /// Parent intent ID (if this refines a previous intent)
    pub parent_id: Option<String>,
}

impl IntentNode {
    pub fn new(agent_id: &str, intent: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            agent_id: agent_id.to_string(),
            timestamp: Utc::now(),
            intent: intent.to_string(),
            provides: Vec::new(),
            requires: Vec::new(),
            constraints: Vec::new(),
            stability: 0.3, // Default: exploring
            evidence: Vec::new(),
            parent_id: None,
        }
    }

    pub fn with_provides(mut self, specs: Vec<InterfaceSpec>) -> Self {
        self.provides = specs;
        self
    }

    pub fn with_requires(mut self, specs: Vec<InterfaceSpec>) -> Self {
        self.requires = specs;
        self
    }

    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_stability(mut self, stability: f64) -> Self {
        self.stability = stability.clamp(0.0, 1.0);
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<Evidence>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }
}

/// A typed interface that an agent provides or requires.
/// This is the unit of compatibility checking between agents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InterfaceSpec {
    /// Unique name for this interface (e.g., "RecipeService.create")
    pub name: String,

    /// The kind of interface
    pub kind: InterfaceKind,

    /// Type signature as a string (e.g., "(recipe: Recipe) -> Recipe")
    pub signature: String,

    /// Module/file path where this interface lives
    pub module_path: String,

    /// Semantic tags for fuzzy matching (e.g., ["crud", "recipe", "create"])
    pub tags: Vec<String>,
}

impl InterfaceSpec {
    pub fn new(name: &str, kind: InterfaceKind, signature: &str) -> Self {
        Self {
            name: name.to_string(),
            kind,
            signature: signature.to_string(),
            module_path: String::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_module(mut self, path: &str) -> Self {
        self.module_path = path.to_string();
        self
    }

    pub fn with_tags(mut self, tags: Vec<&str>) -> Self {
        self.tags = tags.into_iter().map(String::from).collect();
        self
    }

    /// Structural overlap: name overlap or shared tags
    pub fn structurally_overlaps(&self, other: &InterfaceSpec) -> bool {
        if matching::names_overlap(&self.name, &other.name) {
            return true;
        }
        // Check tag overlap — at least 2 shared tags indicates likely overlap
        let shared_tags = self.tags.iter().filter(|t| other.tags.contains(t)).count();
        shared_tags >= 2
    }

    /// Signature compatibility: superset check with type normalization
    pub fn signature_compatible(&self, other: &InterfaceSpec) -> bool {
        matching::signatures_compatible(&self.signature, &other.signature)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterfaceKind {
    Function,
    Class,
    Model,
    Endpoint,
    Migration,
    Config,
}

/// A constraint that an agent's decision imposes on other scopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// What this constraint affects (e.g., "User model", "database schema")
    pub target: String,

    /// The constraint itself (e.g., "must have author_id: UUID")
    pub requirement: String,

    /// How severe is this constraint
    pub severity: ConstraintSeverity,

    /// Tags for matching which agents this affects
    pub affects_tags: Vec<String>,
}

impl Constraint {
    pub fn new(target: &str, requirement: &str) -> Self {
        Self {
            target: target.to_string(),
            requirement: requirement.to_string(),
            severity: ConstraintSeverity::Required,
            affects_tags: Vec::new(),
        }
    }

    pub fn with_severity(mut self, severity: ConstraintSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_affects(mut self, tags: Vec<&str>) -> Self {
        self.affects_tags = tags.into_iter().map(String::from).collect();
        self
    }

    /// Check if this constraint applies to a given intent based on tag overlap
    pub fn applies_to(&self, intent: &IntentNode) -> bool {
        let all_intent_tags: Vec<&str> = intent
            .provides
            .iter()
            .chain(intent.requires.iter())
            .flat_map(|spec| spec.tags.iter().map(|s| s.as_str()))
            .collect();

        self.affects_tags
            .iter()
            .any(|t| all_intent_tags.contains(&t.as_str()))
    }

    /// Check if two constraints conflict (normalized target comparison)
    pub fn conflicts_with(&self, other: &Constraint) -> bool {
        matching::normalize_constraint_target(&self.target)
            == matching::normalize_constraint_target(&other.target)
            && self.requirement != other.requirement
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintSeverity {
    /// Nice to have — other agents should consider but can ignore
    Preferred,
    /// Must comply — violating this will cause integration failures
    Required,
    /// Critical — violating this will cause data loss or security issues
    Critical,
}

/// Evidence that supports or undermines an intent's stability score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub kind: EvidenceKind,
    pub description: String,
    pub timestamp: DateTime<Utc>,
}

impl Evidence {
    pub fn test_pass(description: &str) -> Self {
        Self {
            kind: EvidenceKind::TestPass,
            description: description.to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn code_committed(description: &str) -> Self {
        Self {
            kind: EvidenceKind::CodeCommitted,
            description: description.to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn consumed_by(agent_id: &str) -> Self {
        Self {
            kind: EvidenceKind::ConsumedByOther,
            description: format!("Consumed by agent {}", agent_id),
            timestamp: Utc::now(),
        }
    }

    pub fn conflict(description: &str) -> Self {
        Self {
            kind: EvidenceKind::Conflict,
            description: description.to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn manual_approval() -> Self {
        Self {
            kind: EvidenceKind::ManualApproval,
            description: "Manually approved".to_string(),
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvidenceKind {
    TestPass,
    TestFail,
    CodeCommitted,
    ConsumedByOther,
    Conflict,
    ManualApproval,
}

/// Result of resolving an agent's intent against the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    pub original_intent: String,
    pub adjustments: Vec<Adjustment>,
    pub conflicts: Vec<ConflictReport>,
    pub adopted_constraints: Vec<Constraint>,
}

impl ResolutionResult {
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    pub fn has_adjustments(&self) -> bool {
        !self.adjustments.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjustment {
    pub kind: AdjustmentKind,
    pub description: String,
    pub source_intent_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AdjustmentKind {
    /// Drop a provision and consume another agent's instead
    ConsumeInstead,
    /// Adopt a constraint from another agent
    AdoptConstraint,
    /// Yield to a higher-stability conflicting decision
    YieldTo,
    /// Modify interface signature for compatibility
    AdaptSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub my_intent_id: String,
    pub their_intent_id: String,
    pub description: String,
    pub their_stability: f64,
    pub resolution_suggestion: String,
}
