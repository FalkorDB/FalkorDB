//! The index changes a unit of work produces, collected by the graph's write methods and
//! published by the unit's owner.

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

#[cfg(feature = "index-falkordb")]
use super::falkordb::pending_writes::PendingIndexWrites;

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
}
