//! The index changes a unit of work produces, collected by the graph's write methods and
//! published by the unit's owner.

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

#[cfg(feature = "index-falkordb")]
use super::falkordb::{cch::CchDirty, pending_writes::PendingIndexWrites};

/// Edge documents to remove, keyed by relationship type: `type_id -> { edge_id
/// -> (src, dst) }`.
///
/// The endpoints ride along because they are captured at deletion time — the
/// edge is gone from the tensor by the time the 24-byte RediSearch key has to
/// be rebuilt, so they cannot be looked up again.
pub type EdgeDocRemovals = FxHashMap<u64, FxHashMap<u64, (u64, u64)>>;

/// The index changes a unit of work produced: RediSearch documents in both directions and for
/// both entity kinds, and, with the `index-falkordb` feature, the native index's staged tuples.
///
/// One type rather than parallel maps, and used for both the per-commit set and the deferred
/// one it folds into. They have to stay in correspondence — a document added to the wrong half
/// is a stale RediSearch entry that nothing later notices — and loose fields on `Pending` made
/// that the caller's job at every site. The graph's write methods take the whole set and fill
/// the part they produce.
#[derive(Default)]
pub struct IndexDocs {
    /// Node documents to add, keyed by label id.
    pub node_adds: FxHashMap<u64, RoaringTreemap>,
    /// Node documents to remove, keyed by label id.
    pub node_removes: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to add, keyed by relationship type id.
    pub edge_adds: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to remove — see [`EdgeDocRemovals`].
    pub edge_removes: EdgeDocRemovals,
    /// The native index's staged changes. Applied by the owner of the unit of work, to the
    /// private version they were staged against, before the RediSearch documents are published;
    /// RediSearch's own folding and publishing ([`absorb`](Self::absorb), `commit`) never touch
    /// it.
    #[cfg(feature = "index-falkordb")]
    pub native: PendingIndexWrites,
    /// The unit's CCH change set (per held index: the node pairs it updated, and whether the
    /// hierarchy can absorb them or must be rebuilt), recorded by the same write methods as
    /// [`native`](Self::native) and flushed by the owner into updated indexes on the private
    /// version once the unit's graph writes are all in
    /// ([`Graph::apply_cch_writes`](crate::graph::graph::Graph::apply_cch_writes)). Empty for a
    /// graph with no CCH index.
    #[cfg(feature = "index-falkordb")]
    pub(crate) cch: CchDirty,
    /// The unit replays writes a primary already accepted: a `GRAPH.EFFECT` buffer, from a
    /// replica's stream or the AOF ([`apply_effects`](crate::effects::v3::apply::apply_effects)).
    /// Such a write is applied, never refused by a rule of this server's own: refusing it
    /// would fail the buffer (a full resync on a replica, `exit(1)` while loading the AOF)
    /// over a rule the primary already applied. Constraints are not checked on a replay
    /// either: they are enforced by the query runtime, which a replay does not run. Of the
    /// checks the graph's write methods make themselves, only the CCH weight check reads it
    /// (`cch::maint::check_edge_weights`).
    pub replay: bool,
}
