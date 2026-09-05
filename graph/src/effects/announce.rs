//! What happened, as the emitter is told it — before any version has a say.
//!
//! These describe a mutation, not a wire: a constraint's type and status, an
//! index's fields and options, how many schema entries existed beforehand.
//! Nothing here is a width, a tag or an opcode. Which is why they sit beside
//! the format traits rather than under a version — the *encoding* of an
//! announced index is v3's, the announced index itself is not, and a v4 will
//! encode these same three types differently rather than needing three of its
//! own.
//!
//! They were under `v3::emit`, which meant a caller describing a constraint had
//! to name a wire version to do it.

use std::sync::Arc;

use crate::{
    entity_type::EntityType,
    graph::constraint::{ConstraintStatus, ConstraintType},
    graph::graph::Graph,
    index::IndexType,
    runtime::value::Value,
};

/// How many schemas and attributes a graph held before some mutation.
///
/// Records carry bare ids, so a buffer is only meaningful to a replica that
/// numbers its dictionaries the same way. Anything registered after this
/// baseline has to be announced in the same buffer, ahead of the records that
/// reference it.
pub struct SchemaBaseline {
    pub labels: usize,
    pub types: usize,
    pub attrs: usize,
}

impl SchemaBaseline {
    /// The counts as they stand now.
    #[must_use]
    pub fn of(g: &Graph) -> Self {
        Self {
            labels: g.get_labels().len(),
            types: g.get_types().len(),
            attrs: g.get_node_attribute_names().len(),
        }
    }
}

/// One constraint statement, and the status this node reached on it.
///
/// Announced twice when validation runs asynchronously — once UNDER
/// CONSTRUCTION, once with whatever it settled on — so the same constraint
/// arrives with a different `status`. The apply side upserts, so the second
/// announcement converges on the first rather than duplicating it.
pub struct AnnouncedConstraint<'a> {
    pub ct: ConstraintType,
    pub entity_type: EntityType,
    /// The status this node reached, and `None` for a drop — which carries no
    /// status field at all.
    pub status: Option<ConstraintStatus>,
    pub label: &'a str,
    pub properties: &'a [Arc<String>],
}

/// One index DDL statement, as the runtime evaluated it.
///
/// `options` is the raw `OPTIONS {...}` map rather than the parsed
/// `IndexOptions`: v2 could not encode it at all and forced the whole statement
/// to replicate as a verbatim query, whereas v3 puts the map on the wire so the
/// replica rebuilds exactly what the master built instead of approximating it.
pub struct AnnouncedIndex<'a> {
    pub entity_type: EntityType,
    pub index_type: &'a IndexType,
    pub label: &'a str,
    pub fields: &'a [Arc<String>],
    pub options: Option<&'a Value>,
}
