use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity_type::EntityType;

/// Type of constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// async validation to refer to a constraint after releasing the read lock.
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
        self.ct == *ct
            && self.entity_type == *entity_type
            && self.label.as_str() == label
            && self.properties.len() == properties.len()
            && self
                .properties
                .iter()
                .zip(properties.iter())
                .all(|(a, b)| a == b)
    }
}
