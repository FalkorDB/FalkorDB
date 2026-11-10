//! Core index types for property-based graph lookups.
//!
//! This module defines the data structures and logic that sit between the
//! query engine and the RediSearch C library.  Instead of scanning every
//! node or relationship, the optimizer can use an [`Index`] to jump
//! straight to entities that match a given property predicate.
//!
//! # Supported index types
//!
//! | [`IndexType`] | RediSearch fields used       | Typical Cypher predicate         |
//! |---------------|-----------------------------|---------------------------------|
//! | `Range`       | NUMERIC + TAG + GEO         | `n.age > 30`, `n.name = "Bob"` |
//! | `Fulltext`    | FULLTEXT                    | Full-text search queries         |
//! | `Vector`      | VECTOR                      | Vector similarity search         |
//!
//! # Data flow
//!
//! ```text
//!  Cypher query
//!       |
//!       v
//!  Optimizer  -- picks index -->  Indexer (indexer.rs)
//!                                    |
//!                        +-----------+-----------+
//!                        |                       |
//!                   IndexQuery              Document
//!                   (read path)            (write path)
//!                        |                       |
//!                        v                       v
//!                  Index::query()       Index::add_document()
//!                        |                       |
//!                        +----------+------------+
//!                                   |
//!                                   v
//!                         RediSearch C API
//!                       (redisearch/mod.rs)
//! ```
//!
//! # Key types
//!
//! - [`Index`] -- owns a RediSearch index handle and its field definitions.
//! - [`Field`] -- a single indexed property (name + type + optional text options).
//! - [`IndexQuery`] -- describes the predicate to evaluate against the index.
//! - [`Document`] -- wraps a RediSearch document for inserting/updating entities.
//! - [`IndexResultsIter`] -- lazy pull-based iterator over C query results.

pub mod falkordb;
pub mod index_docs;
pub mod indexer;
pub mod redisearch;
pub mod text_index_options;
pub mod vector_index_options;
pub use text_index_options::TextIndexOptions;
pub use vector_index_options::VectorIndexOptions;

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    hash::Hash,
    os::raw::{c_char, c_int, c_void},
    ptr::null_mut,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use parking_lot::Mutex;

use crate::runtime::value::Value;

use redisearch::{
    DEFAULT_BLOCK_SIZE, GC_POLICY_FORK, HNSW_DEFAULT_EPSILON, HNSWParams, REDISEARCH_ADD_REPLACE,
    RSDoc, RSFLDOPT_NONE, RSFLDOPT_TXTNOSTEM, RSFLDOPT_TXTPHONETIC, RSFLDTYPE_FULLTEXT,
    RSFLDTYPE_TAG, RSFLDTYPE_VECTOR, RSIndex, RSResultsIterator, RediSearch_CreateDocument2,
    RediSearch_CreateField, RediSearch_CreateIndex, RediSearch_CreateIndexOptions,
    RediSearch_CreateVecSimNode, RediSearch_DeleteDocument, RediSearch_DocumentAddFieldString,
    RediSearch_DocumentAddFieldVector, RediSearch_DropIndex, RediSearch_FreeDocument,
    RediSearch_FreeIndexOptions, RediSearch_GetResultsIterator, RediSearch_IndexAddDocument,
    RediSearch_IndexClone, RediSearch_IndexOptionsSetGCPolicy, RediSearch_IndexOptionsSetLanguage,
    RediSearch_IndexOptionsSetStopwords, RediSearch_IndexRelease, RediSearch_IterateQuery,
    RediSearch_MemUsage, RediSearch_ResultsIteratorFree, RediSearch_ResultsIteratorGetScore,
    RediSearch_ResultsIteratorNext, RediSearch_TagFieldSetCaseSensitive,
    RediSearch_TagFieldSetSeparator, RediSearch_TextFieldSetWeight,
    RediSearch_VecSimTieredParams_Init, RediSearch_VectorFieldSetParams, VecSimAlgo, VecSimMetric,
    VecSimParams, VecSimType,
};

/// Type of index for a property.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum IndexType {
    /// B-tree range index for numeric/string/geo comparisons
    Range,
    /// Full-text search index with tokenization
    #[default]
    Fulltext,
    /// Vector similarity index
    Vector,
}

#[derive(Debug, Default)]
pub struct Field {
    pub name: CString,
    pub ty: IndexType,
    options: Option<TextIndexOptions>,
    vector_options: Option<VectorIndexOptions>,
}

impl Field {
    #[must_use]
    pub fn new(
        name: CString,
        ty: IndexType,
        options: Option<TextIndexOptions>,
    ) -> Self {
        Self {
            name,
            ty,
            options,
            vector_options: None,
        }
    }

    #[must_use]
    pub fn new_with_vector_options(
        name: CString,
        ty: IndexType,
        vector_options: VectorIndexOptions,
    ) -> Self {
        Self {
            name,
            ty,
            options: None,
            vector_options: Some(vector_options),
        }
    }

    #[must_use]
    pub const fn options(&self) -> Option<&TextIndexOptions> {
        self.options.as_ref()
    }

    #[must_use]
    pub const fn vector_options(&self) -> Option<&VectorIndexOptions> {
        self.vector_options.as_ref()
    }
}

impl PartialEq for Field {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.name == other.name && self.ty == other.ty
    }
}

impl Eq for Field {}

impl Hash for Field {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        self.name.hash(state);
    }
}

pub struct IndexInfo {
    pub label: Arc<String>,
    pub pending: i32,
    pub progress: u64,
    pub total: u64,
    pub fields: HashMap<Arc<String>, Vec<Arc<Field>>>,
    /// Attribute names in insertion order. Mirrors `fields.keys()` but
    /// preserves the order the user added them, since `HashMap` iteration
    /// is non-deterministic and `CALL db.indexes()` must surface
    /// properties in the order they were declared for parity with edge.
    pub field_order: Vec<Arc<String>>,
    pub language: Option<Arc<String>>,
    pub stopwords: Option<Vec<Arc<String>>>,
    pub entity_type: String,
}

#[derive(Debug)]
pub enum IndexQuery<T> {
    Equal {
        key: Arc<String>,
        value: T,
    },
    Range {
        key: Arc<String>,
        min: Option<T>,
        max: Option<T>,
        include_min: bool,
        include_max: bool,
    },
    And(Vec<Self>),
    Or(Vec<Self>),
    Point {
        key: Arc<String>,
        point: T,
        radius: T,
    },
    /// IN list query: `property IN [v1, v2, ...]`
    /// The list expression is evaluated at runtime and expanded to `Or(Equal(...), ...)`.
    InList {
        key: Arc<String>,
        list: T,
    },
    /// Array contains query: `value IN property` where property holds an array.
    /// Queries the array sub-fields (numeric:arr or string:arr) based on value type.
    ArrayContains {
        key: Arc<String>,
        value: T,
    },
}

impl IndexQuery<Value> {
    /// Whether an entity whose attributes `get` reads satisfies this query, by the same rules the
    /// `Filter` it was pushed down from applies: Cypher `=`, `<`/`<=`, `IN` and `distance()`. A
    /// comparison that is `null` (a missing attribute, disjoint types, NaN) does not hold.
    ///
    /// This is the scan an index scan falls back to when the native index declines a query's
    /// shape, or when a value is one no index lookup answers exactly (a date, an integer past
    /// 2^53): the planner may have dropped the predicate it pushed, so the scan has to check it,
    /// and it must keep exactly the rows the filter would have kept. A `Point` query holds within its
    /// radius, inclusive: the planner keeps the `distance()` filter above every point scan, which
    /// applies `<` itself.
    pub fn holds(
        &self,
        get: &mut impl FnMut(&Arc<String>) -> Option<Value>,
    ) -> bool {
        use crate::runtime::eval::{all_equals, list_contains};
        use crate::runtime::value::{CompareValue, DisjointOrNull};
        use std::cmp::Ordering;

        let is_true = |v: Result<Value, String>| matches!(v, Ok(Value::Bool(true)));
        let ordered = |a: &Value, b: &Value| match a.compare_value(b) {
            (ord, DisjointOrNull::None) => Some(ord),
            _ => None,
        };
        match self {
            Self::Equal { key, value } => get(key)
                .is_some_and(|v| is_true(all_equals([Ok(v), Ok(value.clone())].into_iter()))),
            Self::Range {
                key,
                min,
                max,
                include_min,
                include_max,
            } => get(key).is_some_and(|v| {
                let above = min.as_ref().is_none_or(|min| {
                    ordered(&v, min).is_some_and(|o| {
                        o == Ordering::Greater || (*include_min && o == Ordering::Equal)
                    })
                });
                let below = max.as_ref().is_none_or(|max| {
                    ordered(&v, max).is_some_and(|o| {
                        o == Ordering::Less || (*include_max && o == Ordering::Equal)
                    })
                });
                above && below
            }),
            Self::And(children) => children.iter().all(|c| c.holds(get)),
            Self::Or(children) => children.iter().any(|c| c.holds(get)),
            Self::Point { key, point, radius } => {
                let (Some(Value::Point(v)), Value::Point(point)) = (get(key), point) else {
                    return false;
                };
                let radius = match radius {
                    Value::Int(r) => *r as f64,
                    Value::Float(r) => *r,
                    _ => return false,
                };
                v.distance(point) <= radius
            }
            Self::InList { key, list } => get(key).is_some_and(|v| is_true(list_contains(list, v))),
            // A property that is not a list is kept: `value IN n.prop` over a scalar is a type
            // error, which the planner's retained filter (it always keeps this one) must raise.
            Self::ArrayContains { key, value } => match get(key) {
                Some(list @ Value::List(_)) => is_true(list_contains(&list, value.clone())),
                Some(Value::Null) | None => false,
                Some(_) => true,
            },
        }
    }
}

/// Lazy iterator over RediSearch query results.
///
/// Wraps the C `RSResultsIterator` and calls `RediSearch_ResultsIteratorNext`
/// on each `.next()`. Frees the C iterator on `Drop`.
///
/// The mapper function `F` extracts the desired item type from each raw
/// iterator step (e.g. just the ID, or ID + score).
pub struct IndexResultsIter<T, F: FnMut(*mut RSResultsIterator, u64) -> T> {
    iter: *mut RSResultsIterator,
    /// Own a strong reference so the spec stays alive for the whole iteration,
    /// even if a concurrent `DROP INDEX` runs. `None` for an empty iterator.
    index: Option<OwnedIndex>,
    map: F,
}

impl<T, F: FnMut(*mut RSResultsIterator, u64) -> T> IndexResultsIter<T, F> {
    const fn new(
        iter: *mut RSResultsIterator,
        index: Option<OwnedIndex>,
        map: F,
    ) -> Self {
        Self { iter, index, map }
    }
}

impl IndexResultsIter<(u64, f64), fn(*mut RSResultsIterator, u64) -> (u64, f64)> {
    #[must_use]
    pub fn empty_scored() -> Self {
        Self {
            iter: null_mut(),
            index: None,
            map: |_, id| (id, 0.0),
        }
    }
}

impl<T, F: FnMut(*mut RSResultsIterator, u64) -> T> Iterator for IndexResultsIter<T, F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter.is_null() {
            return None;
        }
        unsafe {
            let rs_idx = self.index.as_ref().map_or(null_mut(), OwnedIndex::as_ptr);
            loop {
                let mut key_len: usize = 0;
                let key = RediSearch_ResultsIteratorNext(self.iter, rs_idx, &raw mut key_len)
                    .cast::<u8>();
                if key.is_null() {
                    return None;
                }
                // `Document::new` always writes a NODE_DOC_KEY_LEN-char hex key;
                // skip anything else rather than let `decode_id` over-read.
                if key_len != NODE_DOC_KEY_LEN {
                    debug_assert!(false, "unexpected node index key length {key_len}");
                    continue;
                }
                return Some((self.map)(self.iter, decode_id(key)));
            }
        }
    }
}

impl<T, F: FnMut(*mut RSResultsIterator, u64) -> T> Drop for IndexResultsIter<T, F> {
    fn drop(&mut self) {
        if !self.iter.is_null() {
            unsafe {
                RediSearch_ResultsIteratorFree(self.iter);
            }
        }
    }
}

/// Iterator yielding (entity ID, score) pairs from fulltext index queries.
pub type ScoredIdIter = IndexResultsIter<(u64, f64), fn(*mut RSResultsIterator, u64) -> (u64, f64)>;

/// Iterator yielding `(src, dst, edge_id, score)` tuples from edge
/// fulltext queries, with the
/// relevance score via `RediSearch_ResultsIteratorGetScore`.
pub struct ScoredEdgeTripleIter {
    iter: *mut RSResultsIterator,
    /// Strong reference keeping the spec alive for the iteration's lifetime.
    index: Option<OwnedIndex>,
}

impl ScoredEdgeTripleIter {
    const fn new(
        iter: *mut RSResultsIterator,
        index: Option<OwnedIndex>,
    ) -> Self {
        Self { iter, index }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            iter: null_mut(),
            index: None,
        }
    }
}

impl Iterator for ScoredEdgeTripleIter {
    type Item = (u64, u64, u64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter.is_null() {
            return None;
        }
        unsafe {
            let rs_idx = self.index.as_ref().map_or(null_mut(), OwnedIndex::as_ptr);
            loop {
                let mut key_len: usize = 0;
                let key = RediSearch_ResultsIteratorNext(self.iter, rs_idx, &raw mut key_len)
                    .cast::<u8>();
                if key.is_null() {
                    return None;
                }
                // `Document::new_edge` always writes an EDGE_DOC_KEY_LEN-char hex
                // key; skip anything else rather than let `decode_triple` over-read.
                if key_len != EDGE_DOC_KEY_LEN {
                    debug_assert!(false, "unexpected edge index key length {key_len}");
                    continue;
                }
                let triple = decode_triple(key);
                let score = RediSearch_ResultsIteratorGetScore(self.iter);
                return Some((triple[0], triple[1], triple[2], score));
            }
        }
    }
}

impl Drop for ScoredEdgeTripleIter {
    fn drop(&mut self) {
        if !self.iter.is_null() {
            unsafe {
                RediSearch_ResultsIteratorFree(self.iter);
            }
        }
    }
}

/// `ScoredIdIter` wrapped with the `Arc<ThinVec<f32>>` whose data was
/// passed to `RediSearch_CreateVecSimNode`.
///
/// The C API stores the vector pointer *without copying*, so the underlying
/// `f32` slice must outlive the iterator: HNSW iteration is lazy and may
/// dereference that pointer on every `next()`. Holding the `Arc` here ties
/// the data's lifetime to the iterator.
pub struct VectorScoredIdIter {
    inner: ScoredIdIter,
    _vector_owner: Arc<thin_vec::ThinVec<f32>>,
}

impl VectorScoredIdIter {
    /// Empty iterator that still pins the vector arc — used when the
    /// label has no index.
    #[must_use]
    pub fn empty(vector: Arc<thin_vec::ThinVec<f32>>) -> Self {
        Self {
            inner: IndexResultsIter::empty_scored(),
            _vector_owner: vector,
        }
    }
}

impl Iterator for VectorScoredIdIter {
    type Item = (u64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Edge variant of [`VectorScoredIdIter`]: yields
/// `(src, dst, edge_id, distance)` tuples while pinning the source
/// vector buffer.
pub struct VectorScoredEdgeTripleIter {
    inner: ScoredEdgeTripleIter,
    _vector_owner: Arc<thin_vec::ThinVec<f32>>,
}

impl VectorScoredEdgeTripleIter {
    #[must_use]
    pub const fn empty(vector: Arc<thin_vec::ThinVec<f32>>) -> Self {
        Self {
            inner: ScoredEdgeTripleIter::empty(),
            _vector_owner: vector,
        }
    }
}

impl Iterator for VectorScoredEdgeTripleIter {
    type Item = (u64, u64, u64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

// ---- Document-key hex encoding ----
//
// RediSearch's doc table is a trie whose per-node child count is a `u8`. Our doc
// keys are raw little-endian entity ids, so sequential ids drive 256 distinct
// first bytes into the trie root; the 256th wraps the counter to 0, the trie
// `realloc`s its children buffer to ~nothing, and copying the child pointers
// overflows it (heap-buffer-overflow in `TrieMap_Add`). Encoding keys as
// lowercase hex shrinks the first-byte domain from 256 to 16, keeping the trie
// inside its printable-key contract. Mirrors the C fix in
// FalkorDB/FalkorDB#2021 (`src/index/index_doc_key.c`): a `u64` id -> 16 hex
// chars, a `[src, dst, edge_id]` triple -> 48.
const NODE_DOC_KEY_LEN: usize = std::mem::size_of::<u64>() * 2;
const EDGE_DOC_KEY_LEN: usize = std::mem::size_of::<[u64; 3]>() * 2;
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Hex-encode `bytes` into `out` (high nibble first). `out.len()` must be
/// `bytes.len() * 2`.
fn hex_encode_into(
    bytes: &[u8],
    out: &mut [u8],
) {
    for (i, &b) in bytes.iter().enumerate() {
        out[i * 2] = HEX_DIGITS[(b >> 4) as usize];
        out[i * 2 + 1] = HEX_DIGITS[(b & 0x0f) as usize];
    }
}

const fn hex_nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => 0,
    }
}

/// Decode 16 hex chars at `ptr` into the original `u64` id (inverse of
/// `hex_encode_into`). SAFETY: `ptr` must point to at least 16 readable bytes.
unsafe fn decode_id(ptr: *const u8) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, b) in bytes.iter_mut().enumerate() {
        // SAFETY: caller guarantees `ptr` covers 16 readable bytes.
        *b = unsafe { (hex_nibble(*ptr.add(i * 2)) << 4) | hex_nibble(*ptr.add(i * 2 + 1)) };
    }
    u64::from_le_bytes(bytes)
}

/// Decode 48 hex chars at `ptr` into a `[src, dst, edge_id]` triple. SAFETY:
/// `ptr` must point to at least 48 readable bytes.
unsafe fn decode_triple(ptr: *const u8) -> [u64; 3] {
    // SAFETY: caller guarantees `ptr` covers 48 readable bytes (3 × 16).
    unsafe {
        [
            decode_id(ptr),
            decode_id(ptr.add(16)),
            decode_id(ptr.add(32)),
        ]
    }
}

/// A document to be indexed, wrapping a RediSearch document.
///
/// Owns the underlying `RSDoc`: `Drop` frees it via `RediSearch_FreeDocument`
/// unless it was handed to `add_document`, which sets `consumed` because
/// `RediSearch_IndexAddDocument` takes ownership and frees it internally.
/// Not `Clone` — two owners would double-free (or double-add) the same `RSDoc`.
pub struct Document {
    rs_doc: *mut RSDoc,
    id: u64,
    /// Set once `add_document` has passed `rs_doc` to RediSearch, which then
    /// owns and frees it. Gates `Drop` so we never free a consumed doc.
    consumed: bool,
}

impl Drop for Document {
    fn drop(&mut self) {
        // Only free docs RediSearch never took ownership of (e.g. entities
        // scanned during population that had no indexed field). Consumed docs
        // were already freed inside `RediSearch_IndexAddDocument`.
        if !self.consumed && !self.rs_doc.is_null() {
            // SAFETY: `rs_doc` came from `RediSearch_CreateDocument2` and was
            // not consumed, so this is the matching, single free.
            unsafe { RediSearch_FreeDocument(self.rs_doc) };
        }
    }
}

impl Document {
    #[must_use]
    pub fn new(id: u64) -> Self {
        let mut key = [0u8; NODE_DOC_KEY_LEN];
        hex_encode_into(&id.to_le_bytes(), &mut key);
        Self {
            id,
            consumed: false,
            rs_doc: unsafe {
                let doc = RediSearch_CreateDocument2(
                    key.as_ptr().cast::<c_void>(),
                    NODE_DOC_KEY_LEN,
                    null_mut(),
                    1.0,
                    null_mut(),
                );
                debug_assert!(!doc.is_null(), "Failed to create RediSearch document");
                doc
            },
        }
    }

    /// Build a document keyed by the 24-byte `[src, dst, edge_id]`
    /// triple. Used by edge indexes so query results carry endpoints
    /// directly — no tensor scan needed to materialize `(src, dst)`
    /// after an index hit. Mirrors FalkorDB C's `EdgeIndexKey` in
    /// `src/index/index_edge.c`.
    #[must_use]
    pub fn new_edge(
        src: u64,
        dst: u64,
        edge_id: u64,
    ) -> Self {
        let mut key = [0u8; EDGE_DOC_KEY_LEN];
        hex_encode_into(&src.to_le_bytes(), &mut key[0..16]);
        hex_encode_into(&dst.to_le_bytes(), &mut key[16..32]);
        hex_encode_into(&edge_id.to_le_bytes(), &mut key[32..48]);
        Self {
            id: edge_id,
            consumed: false,
            rs_doc: unsafe {
                let doc = RediSearch_CreateDocument2(
                    key.as_ptr().cast::<c_void>(),
                    EDGE_DOC_KEY_LEN,
                    null_mut(),
                    1.0,
                    null_mut(),
                );
                debug_assert!(!doc.is_null(), "Failed to create RediSearch document");
                doc
            },
        }
    }

    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn set(
        &mut self,
        field: &Field,
        value: &Value,
    ) {
        unsafe {
            // Vector fields only accept VecF32 values of the index's dimension;
            // skip everything else, including a field whose dimension is unknown
            // (no vector options, or dimension 0): such a field is created in
            // RediSearch without vector params. RediSearch rejects a whole
            // document whose vector it cannot index, so adding it would drop the
            // entity from every other index on the label too (C skips it the
            // same way: `index.c` "vector dimension mis-match, can't index this
            // vector").
            if field.ty == IndexType::Vector {
                if let Value::VecF32(vec) = value
                    && field
                        .vector_options
                        .as_ref()
                        .is_some_and(|o| o.dimension != 0 && o.dimension == vec.len() as u64)
                {
                    RediSearch_DocumentAddFieldVector(
                        self.rs_doc,
                        field.name.as_ptr().cast::<c_char>(),
                        vec.as_ptr().cast::<c_char>(),
                        vec.len() * std::mem::size_of::<f32>(),
                    );
                }
                return;
            }
            // Fulltext fields only accept String values. Range fields never get here: the native
            // index holds them, and `redisearch_fields` keeps them out of every document.
            debug_assert!(
                field.ty == IndexType::Fulltext,
                "a Range field reached RediSearch"
            );
            if let Value::String(s) = value {
                RediSearch_DocumentAddFieldString(
                    self.rs_doc,
                    field.name.as_ptr(),
                    s.as_ptr().cast::<c_char>(),
                    s.len(),
                    RSFLDTYPE_FULLTEXT,
                );
            }
        }
    }
}

#[derive(Debug)]
pub struct Index {
    /// Unique identity assigned at construction. Lets background workers
    /// (e.g. `populate_index_batch`) detect when the indexer's entry under
    /// their label has been replaced by a fresh `Index` after a DROP +
    /// CREATE round-trip, so they don't decrement the wrong counter.
    id: u64,
    /// The RediSearch index spec. `None` until `create_rs_index` runs (the
    /// former `null` sentinel). Shared across `Index` generations produced by
    /// [`Index::clone_for_update`]; the creation reference is dropped (via
    /// `RediSearch_DropIndex` in [`SpecHandle::drop`]) when the last
    /// generation holding it goes away.
    spec: Option<Arc<SpecHandle>>,
    fields: HashMap<Arc<String>, Vec<Arc<Field>>>,
    /// Attribute keys in insertion order. Tracked alongside `fields` so
    /// `CALL db.indexes()` can return `properties` in declaration order.
    field_order: Vec<Arc<String>>,
    /// Shared across `clone_for_update` generations so population tickets
    /// acquired against an older published generation are released against
    /// the same counters.
    pending_slots: Arc<Mutex<PendingSlots>>,
    progress: AtomicU64,
    total: AtomicU64,
    language: Option<Arc<String>>,
    stopwords: Option<Vec<Arc<String>>>,
}

#[derive(Debug, Default)]
struct PendingSlots {
    /// Generation currently stored in `Index::id`.
    current_generation: u64,
    /// Pending tickets for `current_generation`.
    current_pending: i32,
    /// Pending tickets for older generations aggregated together.
    stale_pending: i32,
}

/// RAII handle for a RediSearch index spec (`RefManager`). Owns one strong
/// reference; `Drop` releases it. The raw `*mut RSIndex` never escapes
/// this type except as a transient passed straight into an FFI call.
#[derive(Debug)]
struct OwnedIndex(std::ptr::NonNull<RSIndex>);

// SAFETY: a RediSearch index spec is a refcounted, internally-synchronized
// `RefManager`; cloning/releasing and using the handle across threads is sound.
unsafe impl Send for OwnedIndex {}
unsafe impl Sync for OwnedIndex {}

impl OwnedIndex {
    /// Wrap an owned strong reference (from `RediSearch_CreateIndex` or
    /// `RediSearch_IndexClone`). `None` for a null/invalidated handle.
    fn from_owned(p: *mut RSIndex) -> Option<Self> {
        std::ptr::NonNull::new(p).map(Self)
    }

    /// Raw handle for an FFI call. The returned pointer must not be stored.
    const fn as_ptr(&self) -> *mut RSIndex {
        self.0.as_ptr()
    }

    /// Acquire an additional strong reference; `None` if the spec was
    /// invalidated by a concurrent `DROP INDEX`.
    fn try_clone(&self) -> Option<Self> {
        Self::from_owned(unsafe { RediSearch_IndexClone(self.as_ptr()) })
    }

    /// Consume the handle, returning the raw pointer without releasing the
    /// reference — for the creation ref, which `RediSearch_DropIndex` both
    /// invalidates and releases in one call.
    const fn into_raw(self) -> *mut RSIndex {
        let p = self.0.as_ptr();
        std::mem::forget(self);
        p
    }
}

impl Drop for OwnedIndex {
    fn drop(&mut self) {
        // No global lock, even for the last release: `IndexSpec_Free` touches no
        // Redis timer (it asserts the temporary-index timer is already stopped, and
        // the ForkGC timer self-terminates by not rescheduling — RediSearch `gc.c`).
        // Only index *creation* needs it, via `RM_CreateTimer`.
        //
        // That is load-bearing for #726: an index-scan iterator drops its cloned
        // handle mid-execute while holding the per-graph read lock, so acquiring the
        // global lock here would be the L1→global inversion.
        unsafe {
            RediSearch_IndexRelease(self.0.as_ptr());
        }
    }
}

/// Shared owner of the *creation* reference to a RediSearch index spec.
///
/// Wrapped in an `Arc` and shared across `Index` generations produced by
/// [`Index::clone_for_update`], so replacing the published `Index` (add
/// field, progress bump, ...) does not drop the live spec. When the last
/// generation releases its `Arc`, `Drop` runs `RediSearch_DropIndex`, which
/// both invalidates the spec and releases the creation ref (outstanding
/// `OwnedIndex` clones held by in-flight read iterators keep the memory
/// alive until they are released).
#[derive(Debug)]
struct SpecHandle(std::mem::ManuallyDrop<OwnedIndex>);

impl SpecHandle {
    const fn new(owned: OwnedIndex) -> Self {
        Self(std::mem::ManuallyDrop::new(owned))
    }

    /// Raw handle for an FFI call. The returned pointer must not be stored.
    fn as_ptr(&self) -> *mut RSIndex {
        self.0.as_ptr()
    }

    /// Acquire an additional strong reference; `None` if the spec was
    /// invalidated by a concurrent `DROP INDEX`.
    fn try_clone_ref(&self) -> Option<OwnedIndex> {
        self.0.try_clone()
    }
}

impl Drop for SpecHandle {
    fn drop(&mut self) {
        // No global lock, for the same reason as `OwnedIndex::drop`: nothing here
        // stops a timer. Load-bearing too — a reader holding the per-graph read lock
        // can be the last holder of the spec (a concurrent `drop_index_bg` swaps the
        // index map under it), and taking the global lock there is the #726
        // inversion.
        //
        // `DropIndex` invalidates the spec *and* releases the creation ref, so pass
        // the raw pointer via `into_raw` to avoid a second release from `OwnedIndex`.
        unsafe {
            let owned = std::mem::ManuallyDrop::take(&mut self.0);
            RediSearch_DropIndex(owned.into_raw());
        }
    }
}

impl Default for Index {
    fn default() -> Self {
        static NEXT_INDEX_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_INDEX_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            spec: None,
            fields: HashMap::new(),
            field_order: Vec::new(),
            pending_slots: Arc::new(Mutex::new(PendingSlots {
                current_generation: id,
                current_pending: 0,
                stale_pending: 0,
            })),
            progress: AtomicU64::new(0),
            total: AtomicU64::new(0),
            language: None,
            stopwords: None,
        }
    }
}

impl Index {
    /// Process-unique identity for this `Index` instance. Used by
    /// background workers to verify the indexer's entry hasn't been
    /// replaced under their label before mutating its counters.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Assign a fresh id to this `Index`. Called when the underlying
    /// RediSearch IndexSpec is rebuilt from scratch (`recreate_index`),
    /// so background workers spawned against the previous incarnation
    /// observe the change and bail out instead of writing partial-spec
    /// docs into the freshly recreated index.
    pub fn bump_id(&mut self) {
        static NEXT_INDEX_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(u64::MAX / 2);
        self.id = NEXT_INDEX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut slots = self.pending_slots.lock();
        // Rotate slots on recreation: previous current work becomes stale.
        slots.stale_pending += slots.current_pending;
        slots.current_generation = self.id;
        slots.current_pending = 0;
    }

    /// Copy this `Index` for a clone-and-swap schema update.
    ///
    /// The copy shares the RediSearch spec handle and the pending-ticket
    /// counters with `self` (so tickets acquired against the currently
    /// published generation stay balanced), while the Rust-side metadata
    /// (fields, language, ...) is deep-cloned so the copy can be mutated
    /// freely before being published.
    #[must_use]
    pub fn clone_for_update(&self) -> Self {
        Self {
            id: self.id,
            spec: self.spec.clone(),
            fields: self.fields.clone(),
            field_order: self.field_order.clone(),
            pending_slots: self.pending_slots.clone(),
            progress: AtomicU64::new(self.progress.load(Ordering::Relaxed)),
            total: AtomicU64::new(self.total.load(Ordering::Relaxed)),
            language: self.language.clone(),
            stopwords: self.stopwords.clone(),
        }
    }
}

impl Index {
    // --- RediSearch index lifecycle ---

    /// Returns true if a RediSearch index has been created.
    #[must_use]
    pub const fn has_rs_index(&self) -> bool {
        self.spec.is_some()
    }

    /// Raw spec handle for an FFI call (null if no index yet). Transient — the
    /// returned pointer is passed straight to a `RediSearch_*` call and never
    /// stored. Valid only while `&self` (and thus the owned reference) lives.
    fn rs_ptr(&self) -> *mut RSIndex {
        self.spec.as_ref().map_or(null_mut(), |h| h.as_ptr())
    }

    /// Create the underlying RediSearch index with the given options.
    /// Should only be called when `!self.has_rs_index()`.
    pub fn create_rs_index(
        &mut self,
        label: &Arc<String>,
        stopwords: Option<&Vec<Arc<String>>>,
        language: Option<&Arc<String>>,
    ) -> Result<(), String> {
        unsafe {
            // Requires the global lock (the host's module GIL):
            // `RediSearch_CreateIndex` registers a GC timer in the host event loop
            // (`GCContext_Start` -> `RM_CreateTimer`). Every caller holds it already —
            // a query escalated to writer mode, or the host's own thread on the
            // RDB-load and replica-`GRAPH.EFFECT` paths — so it is not taken here.
            let options = RediSearch_CreateIndexOptions();
            RediSearch_IndexOptionsSetGCPolicy(options, GC_POLICY_FORK as _);

            // `RediSearch_IndexOptionsSetStopwords` deep-copies via
            // `rm_strdup`, so our CStrings can drop as soon as the call
            // returns.
            if let Some(stop_words) = stopwords {
                let c_stopwords: Vec<CString> = stop_words
                    .iter()
                    .map(|s| CString::new(s.as_str()).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut ptrs: Vec<*const c_char> =
                    c_stopwords.iter().map(|cs| cs.as_ptr()).collect();
                RediSearch_IndexOptionsSetStopwords(
                    options,
                    ptrs.as_mut_ptr(),
                    ptrs.len() as c_int,
                );
            } else {
                RediSearch_IndexOptionsSetStopwords(options, null_mut(), 0);
            }

            // `RediSearch_IndexOptionsSetLanguage`, in contrast, only stores
            // the raw pointer (`options->lang = lang`); `_CreateIndex` later
            // reads it via `RSLanguage_Find` to convert to an enum, so the
            // backing `CString` must outlive that call.
            // (Bug uncovered by ASAN: heap-use-after-free in `RSLanguage_Find`
            // when the inner `c_lang` was dropped at end of its if-let block.)
            let _language_owner: Option<CString> = if let Some(lang) = language {
                let c_lang = CString::new(lang.as_str()).map_err(|e| e.to_string())?;
                if RediSearch_IndexOptionsSetLanguage(options, c_lang.as_ptr()) != 0 {
                    return Err(format!("Language is not supported: {lang}"));
                }
                Some(c_lang)
            } else {
                RediSearch_IndexOptionsSetLanguage(options, null_mut());
                None
            };

            let clabel = CString::new(label.as_str()).map_err(|e| e.to_string())?;

            // GIL is already held at the top of this unsafe block, covering
            // CreateIndex's transitive RM_CreateTimer call.
            self.spec = OwnedIndex::from_owned(RediSearch_CreateIndex(
                clabel.as_ptr().cast::<c_char>(),
                options,
            ))
            .map(|owned| Arc::new(SpecHandle::new(owned)));
            assert!(self.spec.is_some(), "RediSearch_CreateIndex returned null");

            RediSearch_FreeIndexOptions(options);

            // Create the special NONE_INDEXABLE_FIELDS tag field used for
            // attribute-existence queries and reported in db.indexes() info.
            let none_field = CString::new("NONE_INDEXABLE_FIELDS").unwrap();
            let field_id = RediSearch_CreateField(
                self.rs_ptr(),
                none_field.as_ptr(),
                RSFLDTYPE_TAG,
                RSFLDOPT_NONE,
            );
            RediSearch_TagFieldSetSeparator(self.rs_ptr(), field_id, 1 as c_char);
            RediSearch_TagFieldSetCaseSensitive(self.rs_ptr(), field_id, 1);
        }
        Ok(())
    }

    /// Register fields in the RediSearch index. Must be called after `create_rs_index`.
    pub fn register_fields(
        &self,
        fields: &HashMap<Arc<String>, Vec<Arc<Field>>>,
        field_options: Option<&TextIndexOptions>,
    ) -> Result<(), String> {
        unsafe {
            // Requires the global lock, like `create_rs_index` and for the same
            // reason: `RediSearch_CreateField` mutates the spec's field array, which
            // the ForkGC timer callback on the main thread can read concurrently — a
            // heap-corrupting race otherwise (reliably reproducible only in coverage
            // builds, where the window is 10-100× wider).
            for field in fields.values().flat_map(|f| f.iter()) {
                match field.ty {
                    // The native index holds Range fields; RediSearch gets no schema for them.
                    IndexType::Range => {}
                    IndexType::Fulltext => {
                        let mut field_options_flag = RSFLDOPT_NONE;
                        let mut weight = 1.0;
                        let effective_options = field_options.or_else(|| field.options());
                        if let Some(options) = effective_options {
                            weight = options.weight.unwrap_or(1.0);
                            if options.nostem.unwrap_or(false) {
                                field_options_flag |= RSFLDOPT_TXTNOSTEM;
                            }
                            // Any code but empty or C's "no" turns phonetic
                            // matching on. RediSearch takes a flag, not an
                            // algorithm, so which code it is cannot reach it —
                            // see `TextIndexOptions::phonetic`.
                            if options.phonetic_enabled() {
                                field_options_flag |= RSFLDOPT_TXTPHONETIC;
                            }
                        }

                        let field_id = RediSearch_CreateField(
                            self.rs_ptr(),
                            field.name.as_ptr(),
                            RSFLDTYPE_FULLTEXT,
                            field_options_flag,
                        );

                        RediSearch_TextFieldSetWeight(self.rs_ptr(), field_id, weight);
                    }
                    IndexType::Vector => {
                        let field_id = RediSearch_CreateField(
                            self.rs_ptr(),
                            field.name.as_ptr(),
                            RSFLDTYPE_VECTOR,
                            RSFLDOPT_NONE,
                        );

                        if let Some(vopts) = field.vector_options()
                            && vopts.dimension > 0
                        {
                            let metric = match vopts
                                .similarity_function
                                .as_deref()
                                .unwrap_or("euclidean")
                            {
                                "euclidean" => VecSimMetric::L2,
                                "ip" => VecSimMetric::Ip,
                                "cosine" => VecSimMetric::Cosine,
                                other => {
                                    return Err(format!(
                                        "Unknown similarity function '{other}', expected 'euclidean', 'ip', or 'cosine'"
                                    ));
                                }
                            };

                            // RediSearch 8.6 takes a full VecSimParams by value.
                            // FalkorDB builds a TIERED index (a flat brute-force
                            // buffer in front of an HNSW backend, with async
                            // migration), matching C FalkorDB. The HNSW params
                            // are the tiered primary; VecSimTieredParams_Init
                            // fills the tiered runtime fields (worker pool, submit
                            // callback, weak ref, flat-buffer limit, logCtx).
                            let mut primary: VecSimParams = std::mem::zeroed();
                            primary.algo = VecSimAlgo::Hnswlib;
                            primary.algoParams.hnswParams = HNSWParams {
                                type_: VecSimType::Float32,
                                dim: vopts.dimension as usize,
                                metric,
                                multi: false,
                                initialCapacity: 0,
                                // Must be non-zero: the LLAPI path doesn't
                                // normalize blockSize (unlike FT.CREATE), and 0
                                // corrupts the heap on the first HNSW op.
                                blockSize: DEFAULT_BLOCK_SIZE,
                                M: vopts.m.unwrap_or(16),
                                efConstruction: vopts.ef_construction.unwrap_or(200),
                                efRuntime: vopts.ef_runtime.unwrap_or(10),
                                epsilon: HNSW_DEFAULT_EPSILON,
                            };

                            let mut params: VecSimParams = std::mem::zeroed();
                            params.algo = VecSimAlgo::Tiered;
                            // `primary` outlives both calls below: Init reads it,
                            // VectorFieldSetParams deep-copies it.
                            params.algoParams.tieredParams.primaryIndexParams = &raw mut primary;

                            if RediSearch_VecSimTieredParams_Init(
                                &raw mut params,
                                self.rs_ptr(),
                                field.name.as_ptr(),
                            ) != 0
                            {
                                return Err(format!(
                                    "failed to init tiered params for vector field '{}'",
                                    field.name.to_string_lossy()
                                ));
                            }

                            // Returns REDISEARCH_OK (0) on success.
                            if RediSearch_VectorFieldSetParams(
                                self.rs_ptr(),
                                field_id,
                                &raw const params,
                            ) != 0
                            {
                                return Err(format!(
                                    "failed to configure vector field '{}'",
                                    field.name.to_string_lossy()
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if an Int value would lose precision when cast to f64.
    /// Uses the same bitmask as C FalkorDB's RediSearch INT64 workaround,
    /// applied to the value's magnitude so negative integers are handled
    /// correctly.
    #[must_use]
    pub const fn int_loses_f64_precision(i: i64) -> bool {
        i.unsigned_abs() & 0x7FF0_0000_0000_0000 != 0
    }

    /// Execute a fulltext query and return matching entity IDs with scores.
    pub fn fulltext_query(
        &self,
        query: &str,
    ) -> Result<ScoredIdIter, String> {
        let cstr = CString::new(query).map_err(|e| e.to_string())?;
        let Some(index) = self.spec.as_ref().and_then(|h| h.try_clone_ref()) else {
            return Ok(IndexResultsIter::empty_scored());
        };
        let mut err: *mut c_char = null_mut();
        unsafe {
            let iter =
                RediSearch_IterateQuery(index.as_ptr(), cstr.as_ptr(), query.len(), &raw mut err);
            if !err.is_null() {
                let msg = CStr::from_ptr(err).to_string_lossy().into_owned();
                drop(CString::from_raw(err));
                return Err(msg);
            }
            Ok(IndexResultsIter::new(iter, Some(index), |iter, id| {
                let score = RediSearch_ResultsIteratorGetScore(iter);
                (id, score)
            }))
        }
    }

    /// Execute a fulltext query on an *edge* index and return matching
    /// `(src, dst, edge_id, score)` tuples. Mirrors [`fulltext_query`],
    /// but reads the 24-byte `[u64; 3]` key written by
    /// `Document::new_edge` instead of treating the result as an 8-byte
    /// entity id. Should only be called on edge indexes.
    pub fn fulltext_query_edges(
        &self,
        query: &str,
    ) -> Result<ScoredEdgeTripleIter, String> {
        let cstr = CString::new(query).map_err(|e| e.to_string())?;
        let Some(index) = self.spec.as_ref().and_then(|h| h.try_clone_ref()) else {
            return Ok(ScoredEdgeTripleIter::empty());
        };
        let mut err: *mut c_char = null_mut();
        unsafe {
            let iter =
                RediSearch_IterateQuery(index.as_ptr(), cstr.as_ptr(), query.len(), &raw mut err);
            if !err.is_null() {
                let msg = CStr::from_ptr(err).to_string_lossy().into_owned();
                drop(CString::from_raw(err));
                return Err(msg);
            }
            Ok(ScoredEdgeTripleIter::new(iter, Some(index)))
        }
    }

    /// Execute a KNN vector query and yield `(entity_id, score)`
    /// pairs in HNSW iteration order. The yielded `score` is the
    /// raw RediSearch relevance score read via
    /// `RediSearch_ResultsIteratorGetScore` — *not* the metric
    /// distance. Callers that need the actual distance under the
    /// configured similarity function should look up each entity's
    /// vector and compute it themselves
    /// (see [`Graph::vector_query_nodes`]).
    ///
    /// `vector` must be a dense `f32` slice whose length matches the
    /// indexed dimension; `field` is the property name used at index
    /// creation. The iterator returned holds onto the `Arc` so the
    /// underlying buffer outlives lazy HNSW iteration — RediSearch
    /// stores the pointer without copying. Caller is responsible for
    /// using this only on indexes that actually have a vector field
    /// for `field`.
    pub fn vector_query(
        &self,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<VectorScoredIdIter, String> {
        // The field name in the underlying RediSearch index is prefixed
        // with `vector:` (see `Indexer::create_index`). Without this
        // prefix the C-level KNN query silently matches no documents.
        let cstr = CString::new(format!("vector:{field}")).map_err(|e| e.to_string())?;
        // SAFETY: `f32` is `repr(C)` and 4-byte aligned; we expose it as a
        // bag of bytes for RediSearch's KNN API which expects raw float32
        // memory. The `Arc<ThinVec<f32>>` is moved into the returned
        // iterator so the buffer stays valid for the iterator's lifetime.
        let nbytes = std::mem::size_of_val(vector.as_slice());
        let Some(index) = self.spec.as_ref().and_then(|h| h.try_clone_ref()) else {
            return Ok(VectorScoredIdIter {
                inner: IndexResultsIter::empty_scored(),
                _vector_owner: vector,
            });
        };
        unsafe {
            let query_node = RediSearch_CreateVecSimNode(
                index.as_ptr(),
                cstr.as_ptr(),
                vector.as_ptr().cast::<c_char>(),
                nbytes,
                k,
            );
            if query_node.is_null() {
                return Ok(VectorScoredIdIter {
                    inner: IndexResultsIter::empty_scored(),
                    _vector_owner: vector,
                });
            }
            // GetResultsIterator takes ownership of `query_node`: on success
            // the iter owns it (freed by ResultsIteratorFree on Drop); on
            // failure RediSearch already frees `query_node` internally
            // before returning NULL (handleIterCommon → ResultsIteratorFree
            // → QAST_Destroy(qast.root)).
            let iter = RediSearch_GetResultsIterator(query_node, index.as_ptr());
            if iter.is_null() {
                return Ok(VectorScoredIdIter {
                    inner: IndexResultsIter::empty_scored(),
                    _vector_owner: vector,
                });
            }
            Ok(VectorScoredIdIter {
                inner: IndexResultsIter::new(iter, Some(index), |it, id| {
                    let score = RediSearch_ResultsIteratorGetScore(it);
                    (id, score)
                }),
                _vector_owner: vector,
            })
        }
    }

    /// Like [`vector_query`], but for *edge* indexes: yields
    /// `(src, dst, edge_id, score)` tuples — same caveat applies,
    /// `score` is the raw RediSearch score, not the metric distance.
    /// Mirrors [`fulltext_query_edges`] in how the 24-byte `[u64; 3]`
    /// key is read. Should only be called on edge indexes.
    pub fn vector_query_edges(
        &self,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<VectorScoredEdgeTripleIter, String> {
        // See [`vector_query`] — RediSearch field name is `vector:<attr>`.
        let cstr = CString::new(format!("vector:{field}")).map_err(|e| e.to_string())?;
        let nbytes = std::mem::size_of_val(vector.as_slice());
        let Some(index) = self.spec.as_ref().and_then(|h| h.try_clone_ref()) else {
            return Ok(VectorScoredEdgeTripleIter {
                inner: ScoredEdgeTripleIter::empty(),
                _vector_owner: vector,
            });
        };
        unsafe {
            let query_node = RediSearch_CreateVecSimNode(
                index.as_ptr(),
                cstr.as_ptr(),
                vector.as_ptr().cast::<c_char>(),
                nbytes,
                k,
            );
            if query_node.is_null() {
                return Ok(VectorScoredEdgeTripleIter {
                    inner: ScoredEdgeTripleIter::empty(),
                    _vector_owner: vector,
                });
            }
            // See `vector_query` — RediSearch owns and frees `query_node`
            // in both branches (success: iter owns it; failure: freed
            // internally before NULL is returned).
            let iter = RediSearch_GetResultsIterator(query_node, index.as_ptr());
            if iter.is_null() {
                return Ok(VectorScoredEdgeTripleIter {
                    inner: ScoredEdgeTripleIter::empty(),
                    _vector_owner: vector,
                });
            }
            Ok(VectorScoredEdgeTripleIter {
                inner: ScoredEdgeTripleIter::new(iter, Some(index)),
                _vector_owner: vector,
            })
        }
    }

    /// Add a document to the index.
    pub fn add_document(
        &self,
        doc: &mut Document,
    ) {
        // A label whose indexes are all Range has no RediSearch spec: the native index holds them.
        // Nothing is built for it, but a document that gets here anyway is dropped (and freed).
        if self.spec.is_none() {
            return;
        }
        unsafe {
            let res = RediSearch_IndexAddDocument(
                self.rs_ptr(),
                doc.rs_doc,
                REDISEARCH_ADD_REPLACE as c_int,
                null_mut(),
            );
            debug_assert_eq!(res, 0);
        }
        // RediSearch took ownership of `rs_doc` and frees it internally (it
        // `rm_free`s the doc on the success path). Mark consumed unconditionally
        // so `Drop` never frees it again: the only path that returns an error
        // *without* freeing is a rare `NewAddDocumentCtx == NULL` preprocessing
        // failure, and leaking one doc there is far safer than a double-free.
        doc.consumed = true;
    }

    /// Delete a document from the index by entity ID.
    pub fn delete_document(
        &self,
        id: u64,
    ) {
        if self.spec.is_none() {
            return; // no RediSearch spec: nothing was ever added
        }
        let mut key = [0u8; NODE_DOC_KEY_LEN];
        hex_encode_into(&id.to_le_bytes(), &mut key);
        unsafe {
            RediSearch_DeleteDocument(
                self.rs_ptr(),
                key.as_ptr().cast::<c_void>(),
                NODE_DOC_KEY_LEN,
            );
        }
    }

    /// Delete an edge-index document by its 24-byte `[src, dst, edge_id]`
    /// key (set via `Document::new_edge`).
    pub fn delete_edge_document(
        &self,
        src: u64,
        dst: u64,
        edge_id: u64,
    ) {
        if self.spec.is_none() {
            return; // no RediSearch spec: nothing was ever added
        }
        let mut key = [0u8; EDGE_DOC_KEY_LEN];
        hex_encode_into(&src.to_le_bytes(), &mut key[0..16]);
        hex_encode_into(&dst.to_le_bytes(), &mut key[16..32]);
        hex_encode_into(&edge_id.to_le_bytes(), &mut key[32..48]);
        unsafe {
            RediSearch_DeleteDocument(
                self.rs_ptr(),
                key.as_ptr().cast::<c_void>(),
                EDGE_DOC_KEY_LEN,
            );
        }
    }

    // --- fields ---

    /// Check if any field has the Fulltext index type.
    #[must_use]
    pub fn has_fulltext_field(&self) -> bool {
        self.fields
            .values()
            .any(|fields| fields.iter().any(|f| f.ty == IndexType::Fulltext))
    }

    /// Check if a specific attribute is indexed.
    #[must_use]
    pub fn contains_field(
        &self,
        attr: &Arc<String>,
    ) -> bool {
        self.fields.contains_key(attr)
    }

    /// Check if a specific attribute has a field with the given index type.
    #[must_use]
    pub fn has_field_with_type(
        &self,
        attr: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        self.fields
            .get(attr)
            .is_some_and(|fields| fields.iter().any(|f| f.ty == *index_type))
    }

    /// Get all fields for a given attribute.
    #[must_use]
    pub fn get_fields(
        &self,
        attr: &Arc<String>,
    ) -> Option<&Vec<Arc<Field>>> {
        self.fields.get(attr)
    }

    /// Push a field to an existing attribute's field list.
    pub fn add_field_to_existing(
        &mut self,
        attr: &Arc<String>,
        field: Arc<Field>,
    ) {
        if let Some(fields) = self.fields.get_mut(attr) {
            fields.push(field);
        }
    }

    /// Insert a new attribute with its initial field.
    pub fn insert_field(
        &mut self,
        attr: Arc<String>,
        field: Arc<Field>,
    ) {
        if !self.fields.contains_key(&attr) {
            self.field_order.push(attr.clone());
        }
        self.fields.insert(attr, vec![field]);
    }

    /// Remove all fields for an attribute. Returns true if the attr existed.
    pub fn remove_field(
        &mut self,
        attr: &Arc<String>,
    ) -> bool {
        let removed = self.fields.remove(attr).is_some();
        if removed {
            self.field_order.retain(|a| a != attr);
        }
        removed
    }

    /// Retain only fields that don't match the given index type for a specific attribute.
    pub fn retain_fields(
        &mut self,
        attr: &Arc<String>,
        index_type: &IndexType,
    ) {
        if let Some(fields) = self.fields.get_mut(attr) {
            fields.retain(|f| f.ty != *index_type);
        }
    }

    /// Check if the index has no fields at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get all attribute names.
    #[must_use]
    pub fn field_keys(&self) -> Vec<Arc<String>> {
        self.fields.keys().cloned().collect()
    }

    /// Get a reference to all fields.
    #[must_use]
    pub const fn fields(&self) -> &HashMap<Arc<String>, Vec<Arc<Field>>> {
        &self.fields
    }

    /// Attribute keys in insertion order.
    #[must_use]
    pub fn field_order(&self) -> &[Arc<String>] {
        &self.field_order
    }

    /// Iterate over all Field objects (flattened across all attributes).
    pub fn all_fields(&self) -> impl Iterator<Item = &Arc<Field>> {
        self.fields.values().flat_map(|f| f.iter())
    }

    // --- status ---

    /// An index is operational when there are no pending changes.
    #[must_use]
    pub fn is_operational(&self) -> bool {
        self.pending_count() == 0
    }

    /// Set the index population progress.
    pub fn set_progress(
        &self,
        progress: u64,
        total: u64,
    ) {
        self.progress.store(progress, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }

    /// Get the current progress values.
    #[must_use]
    pub fn progress(&self) -> (u64, u64) {
        (
            self.progress.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }

    // --- pending_changes ---

    /// Increment pending counter for a specific generation and return its
    /// previous value.
    pub fn increment_pending_for_generation(
        &self,
        generation_id: u64,
    ) -> i32 {
        let mut slots = self.pending_slots.lock();
        if generation_id == slots.current_generation {
            let prev = slots.current_pending;
            slots.current_pending += 1;
            prev
        } else {
            let prev = slots.stale_pending;
            slots.stale_pending += 1;
            prev
        }
    }

    /// Decrement pending counter for a specific generation only if > 0.
    /// Returns the previous value, or 0 if no decrement happened.
    pub fn try_decrement_pending_for_generation(
        &self,
        generation_id: u64,
    ) -> i32 {
        let mut slots = self.pending_slots.lock();
        if generation_id == slots.current_generation {
            let prev = slots.current_pending;
            if prev > 0 {
                slots.current_pending -= 1;
            }
            prev
        } else {
            let prev = slots.stale_pending;
            if prev > 0 {
                slots.stale_pending -= 1;
            }
            prev
        }
    }

    /// Get pending changes count for a specific generation.
    #[must_use]
    pub fn pending_count_for_generation(
        &self,
        generation_id: u64,
    ) -> i32 {
        let slots = self.pending_slots.lock();
        if generation_id == slots.current_generation {
            slots.current_pending
        } else {
            slots.stale_pending
        }
    }

    /// Get the current pending changes count.
    #[must_use]
    pub fn pending_count(&self) -> i32 {
        self.pending_slots.lock().current_pending
    }

    // --- language ---

    /// Get a reference to the language setting, if any.
    #[must_use]
    pub const fn language(&self) -> Option<&Arc<String>> {
        self.language.as_ref()
    }

    /// Set the language for this index.
    pub fn set_language(
        &mut self,
        language: Option<Arc<String>>,
    ) {
        self.language = language;
    }

    // --- stopwords ---

    /// Get a reference to the stopwords list, if any.
    #[must_use]
    pub const fn stopwords(&self) -> Option<&Vec<Arc<String>>> {
        self.stopwords.as_ref()
    }

    /// Set the stopwords for this index.
    pub fn set_stopwords(
        &mut self,
        stopwords: Option<Vec<Arc<String>>>,
    ) {
        self.stopwords = stopwords;
    }

    // --- memory usage ---

    /// Report memory consumed by the underlying RediSearch index.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.spec
            .as_ref()
            .map_or(0, |index| unsafe { RediSearch_MemUsage(index.as_ptr()) })
    }

    // --- index count ---

    /// Get the number of indexed documents.
    #[must_use]
    pub fn index_count(&self) -> usize {
        self.fields.values().map(Vec::len).sum()
    }

    pub fn recreate_index(
        &mut self,
        label: &Arc<String>,
    ) -> Result<(), String> {
        // Release this generation's reference to the old spec. The actual
        // `RediSearch_DropIndex` (invalidate + release the creation ref)
        // happens in `SpecHandle::drop` once the last holder releases it —
        // a previously published `Index` generation still visible to
        // in-flight readers keeps it alive until then. `create_rs_index`
        // below installs a fresh `self.spec`.
        self.spec = None;
        let stopwords = self.stopwords.clone();
        let language = self.language.clone();
        self.create_rs_index(label, stopwords.as_ref(), language.as_ref())?;
        self.register_fields(self.fields(), None)?;
        // Bump id so any background populate still running against the
        // previous rs_idx detects the recreation and bails out. Pending is
        // tracked per generation, so stale workers release against their own
        // generation counter and never touch this fresh generation.
        self.bump_id();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{EDGE_DOC_KEY_LEN, NODE_DOC_KEY_LEN, decode_id, decode_triple, hex_encode_into};

    /// Encode a node id the way `Document::new` does, then decode it back.
    fn node_roundtrip(id: u64) -> u64 {
        let mut key = [0u8; NODE_DOC_KEY_LEN];
        hex_encode_into(&id.to_le_bytes(), &mut key);
        // SAFETY: `key` holds exactly NODE_DOC_KEY_LEN (16) readable hex bytes.
        unsafe { decode_id(key.as_ptr()) }
    }

    #[test]
    fn node_key_roundtrip() {
        for id in [
            0u64,
            1,
            0xff,
            0x100, // 256 — the boundary that drove the original trie overflow
            0x1234_5678_9abc_def0,
            u64::MAX,
        ] {
            assert_eq!(node_roundtrip(id), id, "roundtrip failed for {id:#x}");
        }
    }

    #[test]
    fn node_key_is_16_lowercase_hex() {
        let mut key = [0u8; NODE_DOC_KEY_LEN];
        hex_encode_into(&0xdead_beef_u64.to_le_bytes(), &mut key);
        assert_eq!(key.len(), 16);
        assert!(
            key.iter()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "key is not all-lowercase hex: {key:?}"
        );
    }

    #[test]
    fn edge_triple_roundtrip() {
        let triple = [0u64, 0x00ff_00ff_00ff_00ff, u64::MAX];
        let mut key = [0u8; EDGE_DOC_KEY_LEN];
        for (i, &v) in triple.iter().enumerate() {
            hex_encode_into(&v.to_le_bytes(), &mut key[i * 16..i * 16 + 16]);
        }
        // SAFETY: `key` holds exactly EDGE_DOC_KEY_LEN (48) readable hex bytes.
        let back = unsafe { decode_triple(key.as_ptr()) };
        assert_eq!(back, triple);
    }
}
