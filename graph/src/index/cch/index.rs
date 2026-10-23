//! The `Graph` API over [`CchIndex`] (#3212).
//!
//! The signatures are fixed here so the DDL, maintenance and procedure changes can be written against them.
//! Every body is `unimplemented!()`: nothing in the engine calls these, so a graph with no
//! CCH index behaves exactly as it did before.

use super::{CchError, CchIndex, CchIndexDef};
use crate::graph::graph::Graph;

impl Graph {
    /// Build a CCH index for `def` and hold it on this version of the graph. Synchronous:
    /// unlike the RediSearch indexes there is no background population pass.
    ///
    /// # Errors
    /// [`CchError::AlreadyExists`] if an index with the same definition is already held,
    /// [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`] for an unresolvable
    /// definition, and whatever [`Cch::build`](super::Cch::build) reports.
    ///
    /// # Panics
    /// Always — unimplemented until #3212. Nothing calls it yet.
    pub fn create_cch_index(
        &mut self,
        _def: CchIndexDef,
    ) -> Result<(), CchError> {
        unimplemented!("Graph API, #3212")
    }

    /// # Errors
    /// [`CchError::NotFound`] if no index with this definition is held.
    ///
    /// # Panics
    /// Always — unimplemented until #3212. Nothing calls it yet.
    pub fn drop_cch_index(
        &mut self,
        _def: &CchIndexDef,
    ) -> Result<(), CchError> {
        unimplemented!("Graph API, #3212")
    }

    /// Every CCH index definition held by this version, in no guaranteed order.
    ///
    /// # Panics
    /// Always — unimplemented until #3212. Nothing calls it yet.
    pub fn cch_indexes(&self) -> impl Iterator<Item = &CchIndexDef> {
        unimplemented!("Graph API, #3212");
        // Unreachable, but it pins the return type: `impl Iterator` needs a concrete type
        // and `unimplemented!()` alone leaves the compiler without one.
        #[allow(unreachable_code)]
        std::iter::empty()
    }

    /// # Panics
    /// Always — unimplemented until #3212. Nothing calls it yet.
    #[must_use]
    pub fn cch_index(
        &self,
        _def: &CchIndexDef,
    ) -> Option<&CchIndex> {
        unimplemented!("Graph API, #3212")
    }

    /// Rebuild every CCH index from the current graph. Used by RDB load (the hierarchy is not
    /// persisted, only the definition) and by maintenance.
    ///
    /// # Errors
    /// Whatever [`Cch::build`](super::Cch::build) reports for any held definition.
    ///
    /// # Panics
    /// Always — unimplemented until #3212. Nothing calls it yet.
    pub fn rebuild_cch_indexes(&mut self) -> Result<(), CchError> {
        unimplemented!("Graph API, #3212")
    }
}
