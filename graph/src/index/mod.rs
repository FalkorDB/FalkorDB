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
    RSFLDTYPE_GEO, RSFLDTYPE_NUMERIC, RSFLDTYPE_TAG, RSFLDTYPE_VECTOR,
    RSGeoDistance_RS_GEO_DISTANCE_M, RSIndex, RSRANGE_INF, RSRANGE_NEG_INF, RSResultsIterator,
    RediSearch_CreateDocument2, RediSearch_CreateEmptyNode, RediSearch_CreateField,
    RediSearch_CreateGeoNode, RediSearch_CreateIndex, RediSearch_CreateIndexOptions,
    RediSearch_CreateIntersectNode, RediSearch_CreateNumericNode, RediSearch_CreateTagLexRangeNode,
    RediSearch_CreateTagNode, RediSearch_CreateTagTokenNode, RediSearch_CreateUnionNode,
    RediSearch_CreateVecSimNode, RediSearch_DeleteDocument, RediSearch_DocumentAddFieldGeo,
    RediSearch_DocumentAddFieldNumber, RediSearch_DocumentAddFieldNumericArray,
    RediSearch_DocumentAddFieldString, RediSearch_DocumentAddFieldStringArray,
    RediSearch_DocumentAddFieldVector, RediSearch_DropIndex, RediSearch_FreeDocument,
    RediSearch_FreeIndexOptions, RediSearch_GetResultsIterator, RediSearch_IndexAddDocument,
    RediSearch_IndexClone, RediSearch_IndexOptionsSetGCPolicy, RediSearch_IndexOptionsSetLanguage,
    RediSearch_IndexOptionsSetStopwords, RediSearch_IndexRelease, RediSearch_IterateQuery,
    RediSearch_MemUsage, RediSearch_QueryNodeAddChild, RediSearch_ResultsIteratorFree,
    RediSearch_ResultsIteratorGetScore, RediSearch_ResultsIteratorNext,
    RediSearch_TagFieldSetCaseSensitive, RediSearch_TagFieldSetSeparator,
    RediSearch_TextFieldSetWeight, RediSearch_VecSimTieredParams_Init,
    RediSearch_VectorFieldSetParams, VecSimAlgo, VecSimMetric, VecSimParams, VecSimType,
};

/// Type of index for a property.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
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
    /// Precomputed name for numeric array sub-field (e.g. "range:attr:numeric:arr").
    /// Only set for Range fields.
    numeric_arr_name: Option<CString>,
    /// Precomputed name for string array sub-field (e.g. "range:attr:string:arr").
    /// Only set for Range fields.
    string_arr_name: Option<CString>,
}

impl Field {
    fn make_arr_names(
        name: &CString,
        ty: &IndexType,
    ) -> (Option<CString>, Option<CString>) {
        if *ty == IndexType::Range {
            let base = name.to_str().unwrap_or("");
            (
                CString::new(format!("{base}:numeric:arr")).ok(),
                CString::new(format!("{base}:string:arr")).ok(),
            )
        } else {
            (None, None)
        }
    }

    #[must_use]
    pub fn new(
        name: CString,
        ty: IndexType,
        options: Option<TextIndexOptions>,
    ) -> Self {
        let (numeric_arr_name, string_arr_name) = Self::make_arr_names(&name, &ty);
        Self {
            name,
            ty,
            options,
            vector_options: None,
            numeric_arr_name,
            string_arr_name,
        }
    }

    #[must_use]
    pub fn new_with_vector_options(
        name: CString,
        ty: IndexType,
        vector_options: VectorIndexOptions,
    ) -> Self {
        let (numeric_arr_name, string_arr_name) = Self::make_arr_names(&name, &ty);
        Self {
            name,
            ty,
            options: None,
            vector_options: Some(vector_options),
            numeric_arr_name,
            string_arr_name,
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

    #[must_use]
    pub const fn numeric_arr_name(&self) -> Option<&CString> {
        self.numeric_arr_name.as_ref()
    }

    #[must_use]
    pub const fn string_arr_name(&self) -> Option<&CString> {
        self.string_arr_name.as_ref()
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
    fn new(
        iter: *mut RSResultsIterator,
        index: Option<OwnedIndex>,
        map: F,
    ) -> Self {
        Self { iter, index, map }
    }
}

impl IndexResultsIter<u64, fn(*mut RSResultsIterator, u64) -> u64> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            iter: null_mut(),
            index: None,
            map: |_, id| id,
        }
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

/// Iterator yielding entity IDs from range/tag/geo index queries.
pub type IdIter = IndexResultsIter<u64, fn(*mut RSResultsIterator, u64) -> u64>;

/// Iterator yielding (entity ID, score) pairs from fulltext index queries.
pub type ScoredIdIter = IndexResultsIter<(u64, f64), fn(*mut RSResultsIterator, u64) -> (u64, f64)>;

/// Iterator yielding `(src, dst, edge_id)` triples from edge-index
/// queries. Reads a 24-byte `[u64; 3]` key per result (as written by
/// `Document::new_edge`).
pub struct EdgeTripleIter {
    iter: *mut RSResultsIterator,
    /// Strong reference keeping the spec alive for the iteration's lifetime.
    index: Option<OwnedIndex>,
}

impl EdgeTripleIter {
    fn new(
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

impl Iterator for EdgeTripleIter {
    type Item = (u64, u64, u64);

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
                return Some(decode_triple(key).into());
            }
        }
    }
}

impl Drop for EdgeTripleIter {
    fn drop(&mut self) {
        if !self.iter.is_null() {
            unsafe {
                RediSearch_ResultsIteratorFree(self.iter);
            }
        }
    }
}

/// Iterator yielding `(src, dst, edge_id, score)` tuples from edge
/// fulltext queries. Like [`EdgeTripleIter`] but also exposes the
/// relevance score via `RediSearch_ResultsIteratorGetScore`.
pub struct ScoredEdgeTripleIter {
    iter: *mut RSResultsIterator,
    /// Strong reference keeping the spec alive for the iteration's lifetime.
    index: Option<OwnedIndex>,
}

impl ScoredEdgeTripleIter {
    fn new(
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
/// passed to `RediSearch_CreateVecSimNode`. The C API stores the vector
/// pointer *without copying*, so the underlying `f32` slice must outlive
/// the iterator: HNSW iteration is lazy and may dereference that
/// pointer on every `next()`. Holding the `Arc` here ties the data's
/// lifetime to the iterator.
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

/// Escape a TAG string value so it survives RediSearch's tag tokenizer and the
/// query-side `tag_strtolower` strip pass, collapsing to a single literal
/// token. Mirrors C FalkorDB's `TagEncode_Lower` (FalkorDB/FalkorDB#2021,
/// `src/index/tag_encode.c`): every byte `<= 0x20` (control bytes, whitespace
/// and the `\1` tag separator), `\` (0x5c, the read-side strip target) and `_`
/// (0x5f, the escape prefix itself) is emitted as `_` followed by two lowercase
/// hex digits; all other bytes pass through verbatim. Applied symmetrically on
/// the index write and query paths so the stored key and the query key match.
/// No decoder exists — the encoded form is only ever the RediSearch index key;
/// the real attribute value is served from FalkorDB's own attribute store.
fn tag_encode_lower(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    for &b in src {
        if b <= 0x20 || b == b'\\' || b == b'_' {
            out.push(b'_');
            out.push(HEX_DIGITS[(b >> 4) as usize]);
            out.push(HEX_DIGITS[(b & 0x0f) as usize]);
        } else {
            out.push(b);
        }
    }
    out
}

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
    /// CStrings for array string elements. RediSearch stores raw pointers
    /// into these during `set`, so they must live until after `add_document`.
    string_arr_values: Vec<CString>,
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
            // not consumed, so this is the matching, single free. The
            // `string_arr_values` CStrings are disjoint Rust allocations freed
            // by the automatic field drop that runs after this body.
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
            string_arr_values: Vec::new(),
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
            string_arr_values: Vec::new(),
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
            // Vector fields only accept VecF32 values; skip everything else.
            if field.ty == IndexType::Vector {
                if let Value::VecF32(vec) = value {
                    RediSearch_DocumentAddFieldVector(
                        self.rs_doc,
                        field.name.as_ptr().cast::<c_char>(),
                        vec.as_ptr().cast::<c_char>(),
                        vec.len() * std::mem::size_of::<f32>(),
                    );
                }
                return;
            }
            // Fulltext fields only accept String values.
            if field.ty == IndexType::Fulltext {
                if let Value::String(s) = value {
                    RediSearch_DocumentAddFieldString(
                        self.rs_doc,
                        field.name.as_ptr(),
                        s.as_ptr().cast::<c_char>(),
                        s.len(),
                        RSFLDTYPE_FULLTEXT,
                    );
                }
                return;
            }
            match value {
                Value::Bool(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        f64::from(*i),
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::Int(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        *i as f64,
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::Float(i) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr(),
                        *i,
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::String(s) => {
                    // Only Range fields reach here; add as TAG, encoded so
                    // tag-special bytes (separator, backslash, whitespace)
                    // survive RediSearch tokenization as a single token.
                    let enc = tag_encode_lower(s.as_bytes());
                    RediSearch_DocumentAddFieldString(
                        self.rs_doc,
                        field.name.as_ptr(),
                        enc.as_ptr().cast::<c_char>(),
                        enc.len(),
                        RSFLDTYPE_TAG,
                    );
                }
                Value::Datetime(ts) | Value::Date(ts) | Value::Time(ts) | Value::Duration(ts) => {
                    RediSearch_DocumentAddFieldNumber(
                        self.rs_doc,
                        field.name.as_ptr().cast::<c_char>(),
                        *ts as f64,
                        RSFLDTYPE_NUMERIC,
                    );
                }
                Value::List(items) => {
                    // Index array elements in separate fields for contains queries.
                    // Numeric elements go to "range:{attr}:numeric:arr",
                    // string elements go to "range:{attr}:string:arr".
                    let mut numerics: Vec<f64> = Vec::new();
                    let mut string_cstrs: Vec<CString> = Vec::new();
                    for item in items.iter() {
                        match item {
                            Value::Bool(b) => numerics.push(f64::from(*b)),
                            Value::Int(i) => numerics.push(*i as f64),
                            Value::Float(f) => numerics.push(*f),
                            Value::String(s) => {
                                // Encode to match the single-string TAG write
                                // path and the query path.
                                if let Ok(cs) = CString::new(tag_encode_lower(s.as_bytes())) {
                                    string_cstrs.push(cs);
                                }
                            }
                            _ => {} // Skip non-indexable types
                        }
                    }
                    if !numerics.is_empty()
                        && let Some(name) = field.numeric_arr_name()
                    {
                        RediSearch_DocumentAddFieldNumericArray(
                            self.rs_doc,
                            name.as_ptr(),
                            numerics.as_ptr(),
                            numerics.len(),
                            RSFLDTYPE_NUMERIC,
                        );
                    }
                    if !string_cstrs.is_empty()
                        && let Some(name) = field.string_arr_name()
                    {
                        let ptrs: Vec<*const c_char> =
                            string_cstrs.iter().map(|cs| cs.as_ptr()).collect();
                        RediSearch_DocumentAddFieldStringArray(
                            self.rs_doc,
                            name.as_ptr(),
                            ptrs.as_ptr(),
                            ptrs.len(),
                            RSFLDTYPE_TAG,
                        );
                        // Keep string content CStrings alive — the pointer
                        // array in RediSearch references them. They'll be
                        // properly freed when the Document is dropped.
                        self.string_arr_values.extend(string_cstrs);
                    }
                }
                Value::VecF32(_) => {} // Only for vector fields
                Value::Point(p) => {
                    RediSearch_DocumentAddFieldGeo(
                        self.rs_doc,
                        field.name.as_ptr().cast::<c_char>(),
                        f64::from(p.latitude),
                        f64::from(p.longitude),
                        RSFLDTYPE_GEO,
                    );
                }
                Value::Null
                | Value::Map(_)
                | Value::Node(_)
                | Value::Relationship(_)
                | Value::Path(_) => unreachable!(),
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
    /// former `null` sentinel). Owns the creation strong reference.
    index: Option<OwnedIndex>,
    fields: HashMap<Arc<String>, Vec<Arc<Field>>>,
    /// Attribute keys in insertion order. Tracked alongside `fields` so
    /// `CALL db.indexes()` can return `properties` in declaration order.
    field_order: Vec<Arc<String>>,
    pending_slots: Mutex<PendingSlots>,
    progress: u64,
    total: u64,
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

/// RAII guard for the Redis module GIL. Constructed by [`GilGuard::acquire`],
/// which returns `None` if any required FFI symbol is unresolved or the
/// context allocation returned null — so callers fall through to the
/// unlocked path rather than panicking in `Drop`.
struct GilGuard(*mut redisearch::redis::RedisModuleCtx);

impl GilGuard {
    /// Acquire the Redis module GIL. All four FFI symbols are checked
    /// up-front so the matching `Drop` impl below can never see a missing
    /// `Unlock`/`FreeThreadSafeContext` (panic in `Drop` would abort).
    unsafe fn acquire() -> Option<Self> {
        unsafe {
            let get = redisearch::redis::RedisModule_GetThreadSafeContext?;
            let lock = redisearch::redis::RedisModule_ThreadSafeContextLock?;
            redisearch::redis::RedisModule_ThreadSafeContextUnlock?;
            redisearch::redis::RedisModule_FreeThreadSafeContext?;
            let ctx = get(std::ptr::null_mut());
            if ctx.is_null() {
                return None;
            }
            lock(ctx);
            Some(Self(ctx))
        }
    }
}

impl Drop for GilGuard {
    fn drop(&mut self) {
        unsafe {
            // Both symbols verified `Some` in `acquire`.
            redisearch::redis::RedisModule_ThreadSafeContextUnlock.unwrap()(self.0);
            redisearch::redis::RedisModule_FreeThreadSafeContext.unwrap()(self.0);
        }
    }
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
    fn as_ptr(&self) -> *mut RSIndex {
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
    fn into_raw(self) -> *mut RSIndex {
        let p = self.0.as_ptr();
        std::mem::forget(self);
        p
    }
}

impl Drop for OwnedIndex {
    fn drop(&mut self) {
        // The final release frees the spec -> IndexSpec_Free -> RM_StopTimer,
        // which mutates Redis timer state; off-main-thread callers must hold
        // the module GIL (mirrors `Index::drop` / `create_rs_index`).
        unsafe {
            let _gil = if crate::thread_id::is_main_thread() {
                None
            } else {
                GilGuard::acquire()
            };
            RediSearch_IndexRelease(self.0.as_ptr());
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        // RediSearch_DropIndex transitively calls RM_StopTimer, which mutates
        // Redis-internal state (the timer rax). The Redis main thread holds
        // the module GIL implicitly during command execution, so off-thread
        // callers must acquire it explicitly. Mirrors the C FalkorDB pattern
        // in `_GraphContext_Free` (FalkorDB/src/graph/graphcontext.c).
        //
        // DropIndex both invalidates the spec and releases the creation ref,
        // so hand it the raw pointer via `into_raw` to avoid a double release
        // from `OwnedIndex`'s own Drop. Outstanding clones (held by in-flight
        // read iterators) keep the spec alive until they are released.
        if let Some(index) = self.index.take() {
            unsafe {
                let _gil = if crate::thread_id::is_main_thread() {
                    None
                } else {
                    GilGuard::acquire()
                };
                RediSearch_DropIndex(index.into_raw());
                // _gil drops here, releasing the GIL if it was acquired.
            }
        }
    }
}

impl Default for Index {
    fn default() -> Self {
        static NEXT_INDEX_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_INDEX_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            index: None,
            fields: HashMap::new(),
            field_order: Vec::new(),
            pending_slots: Mutex::new(PendingSlots {
                current_generation: id,
                current_pending: 0,
                stale_pending: 0,
            }),
            progress: 0,
            total: 0,
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
}

impl Index {
    // --- RediSearch index lifecycle ---

    /// Returns true if a RediSearch index has been created.
    #[must_use]
    pub const fn has_rs_index(&self) -> bool {
        self.index.is_some()
    }

    /// Raw spec handle for an FFI call (null if no index yet). Transient — the
    /// returned pointer is passed straight to a `RediSearch_*` call and never
    /// stored. Valid only while `&self` (and thus the owned reference) lives.
    fn rs_ptr(&self) -> *mut RSIndex {
        self.index.as_ref().map_or(null_mut(), OwnedIndex::as_ptr)
    }

    /// Create the underlying RediSearch index with the given options.
    /// Should only be called when `!self.has_rs_index()`.
    pub fn create_rs_index(
        &mut self,
        label: &Arc<String>,
        stopwords: Option<&Vec<Arc<String>>>,
        language: Option<&Arc<String>>,
    ) -> Result<(), String> {
        // RediSearch_CreateIndex transitively calls RM_CreateTimer (via
        // GCContext_Start), which mutates Redis-internal timer state. Off-thread
        // callers (background populate / write worker) must hold the module GIL.
        unsafe {
            let _gil = if crate::thread_id::is_main_thread() {
                None
            } else {
                GilGuard::acquire()
            };
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
            self.index = OwnedIndex::from_owned(RediSearch_CreateIndex(
                clabel.as_ptr().cast::<c_char>(),
                options,
            ));
            assert!(self.index.is_some(), "RediSearch_CreateIndex returned null");

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
            // `RediSearch_CreateField` modifies the index spec's internal field
            // array (numFields, fields[], inverted-index trie setup). RediSearch's
            // ForkGC timer callback – fired on the Redis main thread – can concurrently
            // read or mutate the same spec during its periodic scan. Without the GIL
            // this is a data race that corrupts heap state and crashes the process.
            // Under coverage instrumentation the race window is 10-100× wider, which
            // is why the crash is reliably reproduced only in coverage builds.
            // Mirrors the GIL guards in `create_rs_index`, `Index::drop`, and
            // `OwnedIndex::drop`.
            let _gil = if crate::thread_id::is_main_thread() {
                None
            } else {
                GilGuard::acquire()
            };
            for field in fields.values().flat_map(|f| f.iter()) {
                match field.ty {
                    IndexType::Range => {
                        let types = RSFLDTYPE_NUMERIC | RSFLDTYPE_GEO | RSFLDTYPE_TAG;
                        let field_id = RediSearch_CreateField(
                            self.rs_ptr(),
                            field.name.as_ptr(),
                            types,
                            RSFLDOPT_NONE,
                        );

                        RediSearch_TagFieldSetSeparator(self.rs_ptr(), field_id, 1 as c_char);
                        RediSearch_TagFieldSetCaseSensitive(self.rs_ptr(), field_id, 1);

                        // Array sub-fields: numeric and string array elements
                        // are indexed in separate fields for array-contains queries.
                        // Names are precomputed on the Field struct.
                        if let Some(numeric_arr_name) = field.numeric_arr_name() {
                            RediSearch_CreateField(
                                self.rs_ptr(),
                                numeric_arr_name.as_ptr(),
                                RSFLDTYPE_NUMERIC,
                                RSFLDOPT_NONE,
                            );
                        }

                        let string_arr_field_id =
                            if let Some(string_arr_name) = field.string_arr_name() {
                                RediSearch_CreateField(
                                    self.rs_ptr(),
                                    string_arr_name.as_ptr(),
                                    RSFLDTYPE_TAG,
                                    RSFLDOPT_NONE,
                                )
                            } else {
                                continue;
                            };
                        RediSearch_TagFieldSetSeparator(
                            self.rs_ptr(),
                            string_arr_field_id,
                            1 as c_char,
                        );
                        RediSearch_TagFieldSetCaseSensitive(self.rs_ptr(), string_arr_field_id, 1);
                    }
                    IndexType::Fulltext => {
                        let mut field_options_flag = RSFLDOPT_NONE;
                        let mut weight = 1.0;
                        let effective_options = field_options.or_else(|| field.options());
                        if let Some(options) = effective_options {
                            weight = options.weight.unwrap_or(1.0);
                            if options.nostem.unwrap_or(false) {
                                field_options_flag |= RSFLDOPT_TXTNOSTEM;
                            }
                            if options.phonetic.unwrap_or(false) {
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
                            if RediSearch_VectorFieldSetParams(self.rs_ptr(), field_id, &params)
                                != 0
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

    /// Build a RediSearch query node from an `IndexQuery`.
    ///
    /// Returns a null pointer if the query references unknown fields or
    /// unsupported value types.
    /// Convert a value to f64 for numeric index queries.
    fn value_to_numeric(value: &Value) -> Option<f64> {
        match value {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(f64::from(*b)),
            _ => None,
        }
    }

    /// Check if an Int value would lose precision when cast to f64.
    /// Uses the same bitmask as C FalkorDB's RediSearch INT64 workaround,
    /// applied to the value's magnitude so negative integers are handled
    /// correctly.
    #[must_use]
    pub const fn int_loses_f64_precision(i: i64) -> bool {
        i.unsigned_abs() & 0x7FF0_0000_0000_0000 != 0
    }

    /// Build a RediSearch numeric range node for numeric values.
    fn build_numeric_range_node(
        &self,
        key: &Arc<String>,
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> *mut redisearch::RSQNode {
        let min_f = match min {
            Some(v) => match Self::value_to_numeric(v) {
                Some(f) => f,
                None => return std::ptr::null_mut(),
            },
            None => RSRANGE_NEG_INF,
        };
        let max_f = match max {
            Some(v) => match Self::value_to_numeric(v) {
                Some(f) => f,
                None => return std::ptr::null_mut(),
            },
            None => RSRANGE_INF,
        };
        let Some(field) = self.fields.get(key).and_then(|f| f.first()) else {
            return std::ptr::null_mut();
        };
        unsafe {
            RediSearch_CreateNumericNode(
                self.rs_ptr(),
                field.name.as_ptr(),
                max_f,
                min_f,
                i32::from(include_max),
                i32::from(include_min),
            )
        }
    }

    /// Build a RediSearch TAG lex range node for string range queries.
    /// Matches C FalkorDB's `_StringRangeToQueryNode` in index_query.c.
    fn build_string_range_node(
        &self,
        key: &Arc<String>,
        min: Option<&str>,
        max: Option<&str>,
        include_min: bool,
        include_max: bool,
    ) -> *mut redisearch::RSQNode {
        let Some(field) = self.fields.get(key).and_then(|f| f.first()) else {
            return std::ptr::null_mut();
        };

        let root = unsafe { RediSearch_CreateTagNode(self.rs_ptr(), field.name.as_ptr()) };

        // If both bounds are equal, use exact match (TagTokenNode)
        if let (Some(lo), Some(hi)) = (min, max)
            && lo == hi
        {
            let Ok(token) = CString::new(tag_encode_lower(lo.as_bytes())) else {
                return std::ptr::null_mut();
            };
            let child = unsafe {
                RediSearch_CreateTagTokenNode(self.rs_ptr(), token.as_ptr().cast::<c_char>())
            };
            unsafe { RediSearch_QueryNodeAddChild(root, child) };
            return root;
        }

        // Lex range: NULL pointer means open bound (infinity)
        let min_cstr = min.and_then(|s| CString::new(tag_encode_lower(s.as_bytes())).ok());
        let max_cstr = max.and_then(|s| CString::new(tag_encode_lower(s.as_bytes())).ok());
        let min_ptr = min_cstr.as_ref().map_or(std::ptr::null(), |cs| cs.as_ptr());
        let max_ptr = max_cstr.as_ref().map_or(std::ptr::null(), |cs| cs.as_ptr());

        let child = unsafe {
            RediSearch_CreateTagLexRangeNode(
                self.rs_ptr(),
                field.name.as_ptr(),
                min_ptr,
                max_ptr,
                i32::from(include_min),
                i32::from(include_max),
            )
        };
        unsafe { RediSearch_QueryNodeAddChild(root, child) };
        root
    }

    fn build_query_node(
        &self,
        query: IndexQuery<Value>,
    ) -> *mut redisearch::RSQNode {
        match query {
            // Numeric equality (Int, Float, Bool)
            IndexQuery::Equal { ref key, ref value } if Self::value_to_numeric(value).is_some() => {
                let d = Self::value_to_numeric(value).unwrap();
                let Some(field) = self.fields.get(key).and_then(|f| f.first()) else {
                    return std::ptr::null_mut();
                };
                unsafe {
                    RediSearch_CreateNumericNode(self.rs_ptr(), field.name.as_ptr(), d, d, 1, 1)
                }
            }
            // String equality
            IndexQuery::Equal {
                key,
                value: Value::String(value),
            } => {
                let Some(field) = self.fields.get(&key).and_then(|f| f.first()) else {
                    return std::ptr::null_mut();
                };
                let root = unsafe { RediSearch_CreateTagNode(self.rs_ptr(), field.name.as_ptr()) };
                let Ok(msg) = CString::new(tag_encode_lower(value.as_bytes())) else {
                    return std::ptr::null_mut();
                };
                let child = unsafe {
                    RediSearch_CreateTagTokenNode(self.rs_ptr(), msg.as_ptr().cast::<c_char>())
                };
                unsafe { RediSearch_QueryNodeAddChild(root, child) };
                root
            }
            // Range queries
            IndexQuery::Range {
                ref key,
                ref min,
                ref max,
                include_min,
                include_max,
            } => {
                // Check if this is a string range
                let is_string_range =
                    matches!(min, Some(Value::String(_))) || matches!(max, Some(Value::String(_)));

                if is_string_range {
                    let min_str = match min {
                        Some(Value::String(s)) => Some(s.as_str()),
                        None => None,
                        _ => return std::ptr::null_mut(),
                    };
                    let max_str = match max {
                        Some(Value::String(s)) => Some(s.as_str()),
                        None => None,
                        _ => return std::ptr::null_mut(),
                    };
                    self.build_string_range_node(key, min_str, max_str, include_min, include_max)
                } else {
                    // Numeric range (Int, Float, Bool)
                    self.build_numeric_range_node(
                        key,
                        min.as_ref(),
                        max.as_ref(),
                        include_min,
                        include_max,
                    )
                }
            }
            IndexQuery::Point {
                key,
                point: Value::Point(point),
                radius,
            } => {
                let r = match radius {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    _ => return std::ptr::null_mut(),
                };
                let Some(field) = self.fields.get(&key).and_then(|f| f.first()) else {
                    return std::ptr::null_mut();
                };
                unsafe {
                    RediSearch_CreateGeoNode(
                        self.rs_ptr(),
                        field.name.as_ptr(),
                        f64::from(point.latitude),
                        f64::from(point.longitude),
                        r,
                        RSGeoDistance_RS_GEO_DISTANCE_M,
                    )
                }
            }
            IndexQuery::And(children) => {
                let intersect = unsafe { RediSearch_CreateIntersectNode(self.rs_ptr(), 0) };
                for child in children {
                    let child_node = self.build_query_node(child);
                    if child_node.is_null() {
                        // If any AND child can't be converted, the whole AND
                        // is unsatisfiable — return null to avoid broadening
                        // the query by silently dropping a conjunct.
                        return std::ptr::null_mut();
                    }
                    unsafe { RediSearch_QueryNodeAddChild(intersect, child_node) };
                }
                intersect
            }
            IndexQuery::Or(children) => {
                if children.is_empty() {
                    return unsafe { RediSearch_CreateEmptyNode(self.rs_ptr()) };
                }
                let union_node = unsafe { RediSearch_CreateUnionNode(self.rs_ptr()) };
                for child in children {
                    let child_node = self.build_query_node(child);
                    if !child_node.is_null() {
                        unsafe { RediSearch_QueryNodeAddChild(union_node, child_node) };
                    }
                }
                union_node
            }
            IndexQuery::InList {
                key,
                list: Value::List(items),
            } => {
                if items.is_empty() {
                    return unsafe { RediSearch_CreateEmptyNode(self.rs_ptr()) };
                }
                let union_node = unsafe { RediSearch_CreateUnionNode(self.rs_ptr()) };
                for item in items.iter() {
                    let child = self.build_query_node(IndexQuery::Equal {
                        key: key.clone(),
                        value: item.clone(),
                    });
                    if !child.is_null() {
                        unsafe { RediSearch_QueryNodeAddChild(union_node, child) };
                    }
                }
                union_node
            }
            IndexQuery::ArrayContains { key, ref value } => {
                let Some(field) = self.fields.get(&key).and_then(|f| f.first()) else {
                    return std::ptr::null_mut();
                };
                match value {
                    Value::Int(i) => {
                        let Some(arr_name) = field.numeric_arr_name() else {
                            return std::ptr::null_mut();
                        };
                        unsafe {
                            RediSearch_CreateNumericNode(
                                self.rs_ptr(),
                                arr_name.as_ptr(),
                                *i as f64,
                                *i as f64,
                                1,
                                1,
                            )
                        }
                    }
                    Value::Float(f) => {
                        let Some(arr_name) = field.numeric_arr_name() else {
                            return std::ptr::null_mut();
                        };
                        unsafe {
                            RediSearch_CreateNumericNode(
                                self.rs_ptr(),
                                arr_name.as_ptr(),
                                *f,
                                *f,
                                1,
                                1,
                            )
                        }
                    }
                    Value::Bool(b) => {
                        let Some(arr_name) = field.numeric_arr_name() else {
                            return std::ptr::null_mut();
                        };
                        unsafe {
                            RediSearch_CreateNumericNode(
                                self.rs_ptr(),
                                arr_name.as_ptr(),
                                f64::from(*b),
                                f64::from(*b),
                                1,
                                1,
                            )
                        }
                    }
                    Value::String(s) => {
                        let Some(arr_name) = field.string_arr_name() else {
                            return std::ptr::null_mut();
                        };
                        let root =
                            unsafe { RediSearch_CreateTagNode(self.rs_ptr(), arr_name.as_ptr()) };
                        let Ok(token) = CString::new(tag_encode_lower(s.as_bytes())) else {
                            return std::ptr::null_mut();
                        };
                        let child = unsafe {
                            RediSearch_CreateTagTokenNode(
                                self.rs_ptr(),
                                token.as_ptr().cast::<c_char>(),
                            )
                        };
                        unsafe { RediSearch_QueryNodeAddChild(root, child) };
                        root
                    }
                    _ => std::ptr::null_mut(),
                }
            }
            _ => std::ptr::null_mut(),
        }
    }

    /// Execute an index query and return matching entity IDs.
    pub fn query(
        &self,
        query: IndexQuery<Value>,
    ) -> IdIter {
        // Clone a strong ref for the iterator so the spec outlives a concurrent
        // DROP INDEX; `None` means the spec is gone/uninitialized.
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
            return IndexResultsIter::empty();
        };
        unsafe {
            let query_node = self.build_query_node(query);
            if query_node.is_null() {
                return IndexResultsIter::empty();
            }
            let iter = RediSearch_GetResultsIterator(query_node, index.as_ptr());
            IndexResultsIter::new(iter, Some(index), |_, id| id)
        }
    }

    /// Execute an edge-index query and yield `(src, dst, edge_id)`
    /// triples. Expects documents stored with `Document::new_edge`
    /// (24-byte key encoding the triple). Caller is responsible for
    /// only calling this on edge indexes.
    pub fn query_edges(
        &self,
        query: IndexQuery<Value>,
    ) -> EdgeTripleIter {
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
            return EdgeTripleIter::empty();
        };
        unsafe {
            let query_node = self.build_query_node(query);
            if query_node.is_null() {
                return EdgeTripleIter::empty();
            }
            let iter = RediSearch_GetResultsIterator(query_node, index.as_ptr());
            EdgeTripleIter::new(iter, Some(index))
        }
    }

    /// Execute a fulltext query and return matching entity IDs with scores.
    pub fn fulltext_query(
        &self,
        query: &str,
    ) -> Result<ScoredIdIter, String> {
        let cstr = CString::new(query).map_err(|e| e.to_string())?;
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
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
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
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
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
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
        let Some(index) = self.index.as_ref().and_then(OwnedIndex::try_clone) else {
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
    pub const fn set_progress(
        &mut self,
        progress: u64,
        total: u64,
    ) {
        self.progress = progress;
        self.total = total;
    }

    /// Get the current progress values.
    #[must_use]
    pub const fn progress(&self) -> (u64, u64) {
        (self.progress, self.total)
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
        match &self.index {
            None => 0,
            Some(index) => unsafe { RediSearch_MemUsage(index.as_ptr()) },
        }
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
        // Drop the old spec (invalidate + release the creation ref) under the
        // GIL when off the main thread, same as `Index::drop`. `create_rs_index`
        // below installs a fresh `self.index`.
        if let Some(index) = self.index.take() {
            unsafe {
                let _gil = if crate::thread_id::is_main_thread() {
                    None
                } else {
                    GilGuard::acquire()
                };
                RediSearch_DropIndex(index.into_raw());
            }
        }
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
    use super::{
        EDGE_DOC_KEY_LEN, NODE_DOC_KEY_LEN, decode_id, decode_triple, hex_encode_into,
        tag_encode_lower,
    };

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

    #[test]
    fn tag_encode_escapes_only_expected_bytes() {
        // Bytes <= 0x20 (controls + space), '\\' (0x5c) and '_' (0x5f) become
        // `_` + two lowercase hex digits; everything else passes through.
        assert_eq!(tag_encode_lower(b"plain"), b"plain");
        assert_eq!(tag_encode_lower(b"a b"), b"a_20b"); // space
        assert_eq!(tag_encode_lower(b"a\\b"), b"a_5cb"); // backslash
        assert_eq!(tag_encode_lower(b"a_b"), b"a_5fb"); // the escape prefix itself
        assert_eq!(tag_encode_lower(&[0x00]), b"_00"); // NUL
        assert_eq!(tag_encode_lower(&[0x09]), b"_09"); // tab
        assert_eq!(tag_encode_lower(&[0x01]), b"_01"); // FalkorDB tag separator
        // Other punctuation (comma 0x2c) and high bytes pass through untouched.
        assert_eq!(tag_encode_lower(b"a,b"), b"a,b");
        assert_eq!(tag_encode_lower(&[0xff]), &[0xff]);
    }

    #[test]
    fn tag_encode_output_has_no_interior_nul() {
        // The encoded form is handed to CString::new, so it must never contain a
        // 0x00 byte (NUL itself is escaped to "_00").
        let enc = tag_encode_lower(&[0x00, b'x', 0x20, 0xff, b'\\']);
        assert!(!enc.contains(&0x00), "encoded form contains NUL: {enc:?}");
    }
}
