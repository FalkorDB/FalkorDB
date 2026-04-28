use std::sync::Arc;

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
            ConstraintType::Unique => write!(f, "UNIQUE"),
            ConstraintType::Mandatory => write!(f, "MANDATORY"),
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
            ConstraintStatus::UnderConstruction => write!(f, "UNDER CONSTRUCTION"),
            ConstraintStatus::Operational => write!(f, "OPERATIONAL"),
            ConstraintStatus::Failed => write!(f, "FAILED"),
        }
    }
}

/// A graph constraint (unique or mandatory) on a label/type and set of properties.
#[derive(Clone, Debug)]
pub struct Constraint {
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
        Self {
            ct,
            entity_type,
            label,
            properties,
            status: ConstraintStatus::UnderConstruction,
        }
    }

    /// Check if this constraint matches the given type, entity type, label and properties.
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
