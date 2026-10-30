//! The logic behind `Graph`'s CCH index API (#3212). The public methods live in `graph.rs`
//! with the rest of the `Graph` API and delegate here.
//!
//! The signatures are fixed here so the DDL, maintenance and procedure changes can be written against them.
//! Every body is `unimplemented!()`: nothing in the engine calls these, so a graph with no
//! CCH index behaves exactly as it did before.

use super::{CchError, CchIndex, CchIndexDef};
use crate::graph::graph::Graph;

/// [`Graph::create_cch_index`].
pub(crate) fn create(
    _g: &mut Graph,
    _def: CchIndexDef,
) -> Result<(), CchError> {
    unimplemented!("Graph API, #3212")
}

/// [`Graph::drop_cch_index`].
pub(crate) fn drop_index(
    _g: &mut Graph,
    _def: &CchIndexDef,
) -> Result<(), CchError> {
    unimplemented!("Graph API, #3212")
}

/// [`Graph::cch_indexes`].
pub(crate) fn defs(_g: &Graph) -> impl Iterator<Item = &CchIndexDef> {
    unimplemented!("Graph API, #3212");
    // Unreachable, but it pins the return type: `impl Iterator` needs a concrete type
    // and `unimplemented!()` alone leaves the compiler without one.
    #[allow(unreachable_code)]
    std::iter::empty()
}

/// [`Graph::cch_index`].
pub(crate) fn find<'g>(
    _g: &'g Graph,
    _def: &CchIndexDef,
) -> Option<&'g CchIndex> {
    unimplemented!("Graph API, #3212")
}

/// [`Graph::rebuild_cch_indexes`].
pub(crate) fn rebuild_all(_g: &mut Graph) -> Result<(), CchError> {
    unimplemented!("Graph API, #3212")
}
