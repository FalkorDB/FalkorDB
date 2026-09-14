use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity_type::EntityType;

/// Type of constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintType {
    Unique,
    Mandatory,
}

impl std::fmt::Display for ConstraintType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Unique => write!(f, "UNIQUE"),
            Self::Mandatory => write!(f, "MANDATORY"),
        }
    }
}

/// Status of a constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintStatus {
    UnderConstruction,
    Operational,
    Failed,
}

impl std::fmt::Display for ConstraintStatus {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::UnderConstruction => write!(f, "UNDER CONSTRUCTION"),
            Self::Operational => write!(f, "OPERATIONAL"),
            Self::Failed => write!(f, "FAILED"),
        }
    }
}

/// A graph constraint (unique or mandatory) on a label/type and set of properties.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// Process-unique identifier. Stable across `Vec::swap_remove`, used by
    /// background validation to refer to a constraint after releasing the read lock.
    pub id: u64,
    pub ct: ConstraintType,
    pub entity_type: EntityType,
    pub label: Arc<String>,
    pub properties: Vec<Arc<String>>,
    pub status: ConstraintStatus,
}

impl Constraint {
    pub fn new(
        ct: ConstraintType,
        entity_type: EntityType,
        label: Arc<String>,
        properties: Vec<Arc<String>>,
    ) -> Self {
        static NEXT_CONSTRAINT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_CONSTRAINT_ID.fetch_add(1, Ordering::Relaxed),
            ct,
            entity_type,
            label,
            properties,
            status: ConstraintStatus::UnderConstruction,
        }
    }

    /// Check if this constraint matches the given type, entity type, label and properties.
    #[must_use]
    pub fn matches(
        &self,
        ct: &ConstraintType,
        entity_type: &EntityType,
        label: &str,
        properties: &[Arc<String>],
    ) -> bool {
        if self.ct != *ct
            || self.entity_type != *entity_type
            || self.label.as_str() != label
            || self.properties.len() != properties.len()
        {
            return false;
        }

        let mut p1: Vec<&str> = self.properties.iter().map(|s| s.as_str()).collect();
        let mut p2: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        p1.sort_unstable();
        p2.sort_unstable();
        p1 == p2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_matches_order_insensitive() {
        let label = Arc::new("P".to_string());
        let p_a = Arc::new("a".to_string());
        let p_b = Arc::new("b".to_string());
        let p_c = Arc::new("c".to_string());

        let c = Constraint::new(
            ConstraintType::Mandatory,
            EntityType::Node,
            label.clone(),
            vec![p_a.clone(), p_b.clone()],
        );

        // Same order matches
        assert!(c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Node,
            "P",
            &[p_a.clone(), p_b.clone()]
        ));

        // Reversed order matches
        assert!(c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Node,
            "P",
            &[p_b.clone(), p_a.clone()]
        ));

        // Different properties do not match
        assert!(!c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Node,
            "P",
            &[p_a.clone(), p_c.clone()]
        ));

        // Different length does not match
        assert!(!c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Node,
            "P",
            &[p_a.clone()]
        ));

        // Different label does not match
        assert!(!c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Node,
            "Q",
            &[p_a.clone(), p_b.clone()]
        ));

        // Different constraint type does not match
        assert!(!c.matches(
            &ConstraintType::Unique,
            &EntityType::Node,
            "P",
            &[p_a.clone(), p_b.clone()]
        ));

        // Different entity type does not match
        assert!(!c.matches(
            &ConstraintType::Mandatory,
            &EntityType::Relationship,
            "P",
            &[p_a.clone(), p_b.clone()]
        ));

        // 3-element permutation matches
        let c3 = Constraint::new(
            ConstraintType::Unique,
            EntityType::Relationship,
            label.clone(),
            vec![p_a.clone(), p_b.clone(), p_c.clone()],
        );
        assert!(c3.matches(
            &ConstraintType::Unique,
            &EntityType::Relationship,
            "P",
            &[p_c.clone(), p_a.clone(), p_b.clone()]
        ));
    }
}
