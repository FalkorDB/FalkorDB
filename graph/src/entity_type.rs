//! Entity type discriminator for graph elements.
//!
//! FalkorDB stores two kinds of entities in the graph: **nodes** and
//! **relationships** (edges). Many operations -- index creation, schema
//! introspection, and query planning -- need to distinguish between
//! these two kinds at runtime.
//!
//! ```text
//!   (Node) --[Relationship]--> (Node)
//! ```
//!
//! [`EntityType`] is the single enum used throughout the crate for this
//! purpose. It appears in the parser (CREATE INDEX), the planner (index
//! scan selection), and the graph storage layer (index maintenance).

/// Entity type that can be indexed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityType {
    /// Index on node properties
    Node,
    /// Index on relationship properties
    Relationship,
}

impl std::fmt::Display for EntityType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Node => write!(f, "NODE"),
            Self::Relationship => write!(f, "RELATIONSHIP"),
        }
    }
}
