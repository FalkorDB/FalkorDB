//! Columnar batch representation for vectorized query execution.
//!
//! A [`Batch`] stores multiple rows (up to [`BATCH_SIZE`] = 1024), enabling
//! operators to amortize per-row dispatch overhead and exploit data locality.
//!
//! ```text
//!  Batch (columnar)
//! ┌────────────────────────────────────────────────────────┐
//! │  columns[0]: NodeIds  [n1, n2, n3, n4]                 │
//! │  columns[1]: Ints     [10, 20, 30, 40]   ← SIMD ops   │
//! │  columns[2]: Values   ["a","b","c","d"]                │
//! │  selection:  Some([0, 2, 3])  ← only these rows active │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Storage Model
//!
//! Rows are stored as typed [`Column`]s indexed by `Variable.id`. A "current
//! row" is the pair `(&Batch, row_idx)`; the expression evaluator reads
//! `batch.value_at(var_id, row_idx)`. Id columns (`NodeIds`/`RelTriples`) stay
//! cheap until a property is actually read.
//!
//! ## Zero-Copy Filtering
//!
//! Instead of removing filtered-out rows, operators set a **selection vector**
//! (`Vec<u16>`) listing active row indices. Downstream operators iterate only
//! the active rows via `active_indices()`.

use crate::graph::graph::{NodeId, RelationshipId};
use crate::planner::IR;
use crate::runtime::bitset::BitSet;
use crate::runtime::row::{Row, RowView};
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{CompareValue, Value};
use orx_tree::{Dyn, NodeIdx};
use std::cmp::Ordering;
use std::marker::PhantomData;

use super::ops::aggregate::AggregateOp;
use super::ops::all_shortest_paths::AllShortestPathsOp;
use super::ops::apply::ApplyOp;
use super::ops::cartesian_product::CartesianProductOp;
use super::ops::commit::CommitOp;
use super::ops::cond_traverse::CondTraverseOp;
use super::ops::cond_var_len_traverse::CondVarLenTraverseOp;
use super::ops::create::CreateOp;
use super::ops::delete::DeleteOp;
use super::ops::distinct::DistinctOp;
use super::ops::edge_by_fulltext_scan::EdgeByFulltextScanOp;
use super::ops::edge_by_index_scan::EdgeByIndexScanOp;
use super::ops::edge_by_vector_scan::EdgeByVectorScanOp;
use super::ops::expand_into::ExpandIntoOp;
use super::ops::filter::FilterOp;
use super::ops::foreach::ForEachOp;
use super::ops::include_pending::IncludePendingOp;
use super::ops::limit::LimitOp;
use super::ops::load_csv::LoadCsvOp;
use super::ops::merge::MergeOp;
use super::ops::node_by_fulltext_scan::NodeByFulltextScanOp;
use super::ops::node_by_id_seek::NodeByIdSeekOp;
use super::ops::node_by_index_scan::NodeByIndexScanOp;
use super::ops::node_by_label_and_id_scan::NodeByLabelAndIdScanOp;
use super::ops::node_by_label_scan::NodeByLabelScanOp;
use super::ops::node_by_vector_scan::NodeByVectorScanOp;
use super::ops::optional::OptionalOp;
use super::ops::or_apply_multiplexer::OrApplyMultiplexerOp;
use super::ops::path_builder::PathBuilderOp;
use super::ops::procedure_call::ProcedureCallOp;
use super::ops::project::ProjectOp;
use super::ops::remove::RemoveOp;
use super::ops::semi_apply::SemiApplyOp;
use super::ops::set::SetOp;
use super::ops::skip::SkipOp;
use super::ops::sort::SortOp;
use super::ops::union::UnionOp;
use super::ops::unwind::UnwindOp;
use super::ops::value_hash_join::ValueHashJoinOp;

/// Maximum number of rows in a single batch. Used by every operator that
/// throttles output to one batch per `next()` call.
pub const BATCH_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// NullBitmap — compact null tracking for typed columns
// ---------------------------------------------------------------------------

/// Compact bitmap tracking which rows in a typed column are null.
/// Bit `i` is set (1) if row `i` is null.
pub struct NullBitmap {
    words: Vec<u64>,
    len: usize,
}

impl NullBitmap {
    /// Creates a bitmap with all bits unset (no nulls).
    #[must_use]
    pub fn none(len: usize) -> Self {
        let num_words = len.div_ceil(64);
        Self {
            words: vec![0u64; num_words],
            len,
        }
    }

    /// Creates a bitmap from a slice of Values, setting bit `i` if `values[i]` is Null.
    #[must_use]
    pub fn from_values(values: &[Value]) -> Self {
        let len = values.len();
        let num_words = len.div_ceil(64);
        let mut words = vec![0u64; num_words];
        for (i, v) in values.iter().enumerate() {
            if matches!(v, Value::Null) {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self { words, len }
    }

    /// Returns true if row `idx` is null.
    #[inline]
    #[must_use]
    pub fn is_null(
        &self,
        idx: usize,
    ) -> bool {
        debug_assert!(idx < self.len);
        (self.words[idx / 64] >> (idx % 64)) & 1 != 0
    }

    /// Returns true if any row is null.
    #[inline]
    #[must_use]
    pub fn any_null(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }
}

/// Classifies a fully-bound stored column into the most specific lossless
/// [`Column`] representation supported by the current batch layout.
///
/// Because [`Batch`] does not yet store a null bitmap alongside typed columns,
/// only fully non-null homogeneous columns can be promoted out of
/// [`Column::Values`]. Nullable or mixed-shape columns remain value-backed so
/// `value_at`/`get` continue to round-trip exactly.
#[must_use]
pub fn classify_stored_column(values: Vec<Value>) -> Column {
    if values.is_empty() {
        return Column::Values(values);
    }
    if values.iter().all(|v| matches!(v, Value::Node(_))) {
        let ids = values
            .into_iter()
            .map(|v| match v {
                Value::Node(id) => id,
                _ => unreachable!("guarded by all() above"),
            })
            .collect();
        return Column::NodeIds(ids);
    }
    if values.iter().all(|v| matches!(v, Value::Relationship(_))) {
        let triples = values
            .into_iter()
            .map(|v| match v {
                Value::Relationship(rel) => rel,
                _ => unreachable!("guarded by all() above"),
            })
            .collect();
        return Column::RelIds(triples);
    }
    if values.iter().all(|v| matches!(v, Value::Int(_))) {
        let ints = values
            .into_iter()
            .map(|v| match v {
                Value::Int(i) => i,
                _ => unreachable!("guarded by all() above"),
            })
            .collect();
        return Column::Ints(ints);
    }
    if values.iter().all(|v| matches!(v, Value::Float(_))) {
        let floats = values
            .into_iter()
            .map(|v| match v {
                Value::Float(f) => f,
                _ => unreachable!("guarded by all() above"),
            })
            .collect();
        return Column::Floats(floats);
    }
    Column::Values(values)
}

/// Classifies a `Vec<Value>` into the most specific typed Column plus a NullBitmap.
///
/// - If all non-null values are `Int`: returns `Column::Ints` (nulls get 0 as placeholder)
/// - If all non-null values are `Int` or `Float`: returns `Column::Floats` (ints promoted)
/// - Otherwise: returns `Column::Values` as-is
#[must_use]
pub fn classify_column(values: Vec<Value>) -> (Column, NullBitmap) {
    let nulls = NullBitmap::from_values(&values);

    let mut all_int = true;
    let mut all_numeric = true;

    for v in &values {
        match v {
            Value::Int(_) | Value::Null => {}
            Value::Float(_) => {
                all_int = false;
            }
            _ => {
                all_int = false;
                all_numeric = false;
                break;
            }
        }
    }

    if all_int {
        let ints: Vec<i64> = values
            .into_iter()
            .map(|v| match v {
                Value::Int(i) => i,
                _ => 0, // null placeholder, bitmap tracks nullness
            })
            .collect();
        (Column::Ints(ints), nulls)
    } else if all_numeric {
        let floats: Vec<f64> = values
            .into_iter()
            .map(|v| match v {
                Value::Int(i) => i as f64,
                Value::Float(f) => f,
                _ => 0.0, // null placeholder
            })
            .collect();
        (Column::Floats(floats), nulls)
    } else {
        (Column::Values(values), nulls)
    }
}

/// A single column of homogeneous values, indexed by row position.
#[derive(Clone)]
pub enum Column {
    /// All values are node IDs (from scan/traverse operators).
    NodeIds(Vec<NodeId>),
    /// All values are relationship IDs.
    RelIds(Vec<RelationshipId>),
    /// All values are 64-bit signed integers.
    Ints(Vec<i64>),
    /// All values are 64-bit floating point numbers.
    Floats(Vec<f64>),
    /// Heterogeneous or complex values (fallback for String, List, Map, etc.).
    Values(Vec<Value>),
    /// Column not bound in this batch (all rows are Null for this variable).
    Unbound,
}

impl Column {
    /// Extracts a single [`Value`] from this column at the given row index.
    #[must_use]
    pub fn get(
        &self,
        row: usize,
    ) -> Value {
        match self {
            Self::NodeIds(ids) => Value::Node(ids[row]),
            Self::RelIds(triples) => Value::Relationship(triples[row]),
            Self::Ints(vals) => Value::Int(vals[row]),
            Self::Floats(vals) => Value::Float(vals[row]),
            Self::Values(vals) => vals[row].clone(),
            Self::Unbound => Value::Null,
        }
    }

    /// Returns the number of rows stored in this column.
    /// For `Unbound`, returns 0 (the batch `len` field is authoritative).
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::NodeIds(v) => v.len(),
            Self::RelIds(v) => v.len(),
            Self::Ints(v) => v.len(),
            Self::Floats(v) => v.len(),
            Self::Values(v) => v.len(),
            Self::Unbound => 0,
        }
    }

    /// Returns true if this column has no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new column by gathering values from this column at the given
    /// indices.
    pub fn gather(
        &self,
        indices: impl Iterator<Item = usize>,
    ) -> Self {
        match self {
            Self::NodeIds(v) => Self::NodeIds(indices.map(|i| v[i]).collect()),
            Self::RelIds(v) => Self::RelIds(indices.map(|i| v[i]).collect()),
            Self::Ints(v) => Self::Ints(indices.map(|i| v[i]).collect()),
            Self::Floats(v) => Self::Floats(indices.map(|i| v[i]).collect()),
            Self::Values(v) => Self::Values(indices.map(|i| v[i].clone()).collect()),
            Self::Unbound => Self::Unbound,
        }
    }
}

/// In-progress column accumulator for a single variable slot.
struct ColumnBuilder {
    /// One entry per row pushed so far (padded with `Null` for rows that did
    /// not carry this slot, so all columns stay row-aligned).
    values: Vec<Value>,
    /// True once any row carried a *value* in this slot (`get_by_id`).
    present: bool,
    /// True once any row had this slot's *bound* bit set (`is_bound_by_id`).
    any_bound: bool,
}

/// Builds a columnar [`Batch`] incrementally, one row at a time, transposing
/// row-shaped bindings into per-variable columns.
///
/// Operators push each row's bindings directly into the builder, which appends
/// into the growing columns. The builder owns its values and is lifetime-free.
///
/// The resulting batch preserves the value-present vs. bound distinction
/// (`value_only`) and `origin_row` correlation lineage.
#[derive(Default)]
pub struct BatchBuilder {
    /// One [`ColumnBuilder`] per variable slot seen so far. Index is `var id`.
    cols: Vec<ColumnBuilder>,
    /// `origin_row` for each pushed row (correlation lineage).
    origins: Vec<u32>,
    /// True once any pushed row had a non-zero `origin_row`.
    any_origin: bool,
    /// Number of rows pushed so far.
    rows: usize,
}

impl BatchBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of rows pushed so far.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows
    }

    /// Returns true if no rows have been pushed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Truncates the builder to keep only the first `n` pushed rows.
    ///
    /// Used by scan/traverse operators that enforce a `record_cap` after
    /// buffering. These operators emit rows of homogeneous column shape (every
    /// row binds the same variable slots), so dropping a suffix of rows never
    /// changes which columns are present; the per-column `present`/`any_bound`
    /// flags therefore remain accurate.
    pub fn truncate(
        &mut self,
        n: usize,
    ) {
        if n >= self.rows {
            return;
        }
        for col in &mut self.cols {
            col.values.truncate(n);
        }
        self.origins.truncate(n);
        self.rows = n;
    }

    /// Appends one row from a [`Row`], transposing its bindings into the
    /// growing columns, taking the pool-free owned row.
    pub fn push_row(
        &mut self,
        row: &Row,
    ) {
        self.push_row_with(row, row.origin_row, &[]);
    }

    /// Appends one row built from `base` plus `extra` bindings, stamping the
    /// row's `origin_row` to `origin`. The `extra` slice carries freshly
    /// produced bindings (e.g. a scanned node id and score) that are layered
    /// on top of the base row without an intermediate `Env`/`Row` clone.
    pub fn push_row_with(
        &mut self,
        base: &Row,
        origin: u32,
        extra: &[(u32, Value)],
    ) {
        let r = self.rows;
        // Highest slot touched by either the base row or the extra bindings.
        let mut n = base.len();
        for (id, _) in extra {
            n = n.max(*id as usize + 1);
        }
        while self.cols.len() < n {
            self.cols.push(ColumnBuilder {
                values: vec![Value::Null; r],
                present: false,
                any_bound: false,
            });
        }
        // Index extra bindings by variable ID to avoid O(n) find() calls.
        // For small extra slices this is negligible; for larger ones it's a win.
        let extra_map: Option<&(u32, Value)> = if extra.len() == 1 {
            // Micro-optimize common case: single extra binding.
            Some(&extra[0])
        } else if extra.len() > 1 {
            // For multiple extras, skip building a full map—instead just avoid
            // the find() per column by checking extra directly in hot loop.
            None
        } else {
            None
        };
        for (id, col) in self.cols.iter_mut().enumerate() {
            let id = id as u32;
            let found_in_extra = if let Some((eid, v)) = extra_map {
                if *eid == id {
                    col.values.push(v.clone());
                    true
                } else {
                    false
                }
            } else {
                // For multiple or zero extras, scan directly (avoids hash overhead).
                extra.iter().any(|(eid, v)| {
                    if *eid == id {
                        col.values.push(v.clone());
                        col.present = true;
                        col.any_bound = true;
                        return true;
                    }
                    false
                })
            };
            if found_in_extra {
                col.present = true;
                col.any_bound = true;
                continue;
            }
            // Not in extra: check base row.
            if let Some(v) = base.get_by_id(id) {
                col.values.push(v.clone());
                if base.is_bound_by_id(id) {
                    col.present = true;
                    col.any_bound = true;
                } else if !matches!(v, Value::Null) {
                    // Value-present-but-unbound slot (e.g. aggregate alias).
                    col.present = true;
                }
            } else {
                col.values.push(Value::Null);
            }
        }
        if origin != 0 {
            self.any_origin = true;
        }
        self.origins.push(origin);
        self.rows += 1;
    }

    /// Appends one row read directly from `batch[row]`, transposing its columns
    /// into the growing builder columns. Columnar equivalent of
    /// `push_row(&BatchRow::new(batch, row).to_owned_row())` that avoids
    /// materializing an intermediate [`Row`]: `value_only` slots are preserved
    /// as value-present-but-unbound and the supplied `origin` is recorded.
    pub fn push_batch_row(
        &mut self,
        batch: &Batch,
        row: usize,
        origin: u32,
    ) {
        let r = self.rows;
        let n = batch.columns.len();
        while self.cols.len() < n {
            self.cols.push(ColumnBuilder {
                values: vec![Value::Null; r],
                present: false,
                any_bound: false,
            });
        }
        for (id, col) in self.cols.iter_mut().enumerate() {
            match batch.columns.get(id) {
                Some(c) if !matches!(c, Column::Unbound) => {
                    let v = c.get(row);
                    if batch.value_only.test(id) {
                        // Value-present-but-unbound slot (e.g. aggregate alias).
                        let is_null = matches!(v, Value::Null);
                        col.values.push(v);
                        if !is_null {
                            col.present = true;
                        }
                    } else {
                        col.values.push(v);
                        col.present = true;
                        col.any_bound = true;
                    }
                }
                _ => col.values.push(Value::Null),
            }
        }
        if origin != 0 {
            self.any_origin = true;
        }
        self.origins.push(origin);
        self.rows += 1;
    }

    /// Appends every active row of `batch` via [`push_batch_row`](Self::push_batch_row),
    /// carrying each row's `origin_row`. Columnar replacement for the
    /// `buffer.extend(batch.active_rows())` idiom used to concatenate a
    /// sub-plan's output into a single materialized batch.
    pub fn push_batch_active(
        &mut self,
        batch: &Batch,
    ) {
        for row in batch.active_indices() {
            self.push_batch_row(batch, row, batch.origin_row(row));
        }
    }

    /// Appends one row formed by overlaying the bound bindings of
    /// `right[right_row]` on top of `left[left_row]`, reading both directly from
    /// columnar batches. Columnar equivalent of
    /// `left_row.clone(); merged.merge(&right_row); push_row(&merged)`:
    /// a slot takes the right value where right is bound, otherwise the left
    /// value (preserving left's `value_only` semantics), and is bound iff bound
    /// in either side.
    pub fn push_merged(
        &mut self,
        left: &Batch,
        left_row: usize,
        right: &Batch,
        right_row: usize,
        origin: u32,
    ) {
        let r = self.rows;
        let n = left.columns.len().max(right.columns.len());
        while self.cols.len() < n {
            self.cols.push(ColumnBuilder {
                values: vec![Value::Null; r],
                present: false,
                any_bound: false,
            });
        }
        for (id, col) in self.cols.iter_mut().enumerate() {
            let vid = id as u32;
            if right.is_bound_at(vid, right_row) {
                col.values
                    .push(right.value_at(vid, right_row).unwrap_or(Value::Null));
                col.present = true;
                col.any_bound = true;
                continue;
            }
            // Right not bound here: fall back to the left base row.
            match left.columns.get(id) {
                Some(c) if !matches!(c, Column::Unbound) => {
                    let v = c.get(left_row);
                    if left.value_only.test(id) {
                        let is_null = matches!(v, Value::Null);
                        col.values.push(v);
                        if !is_null {
                            col.present = true;
                        }
                    } else {
                        col.values.push(v);
                        col.present = true;
                        col.any_bound = true;
                    }
                }
                _ => col.values.push(Value::Null),
            }
        }
        if origin != 0 {
            self.any_origin = true;
        }
        self.origins.push(origin);
        self.rows += 1;
    }

    /// Consumes the builder and produces the final columnar [`Batch`].
    #[must_use]
    pub fn finish<'a>(self) -> Batch<'a> {
        if self.rows == 0 {
            return Batch {
                len: 0,
                selection: None,
                columns: Vec::new(),
                origin_rows: None,
                value_only: BitSet::default(),
                _marker: PhantomData,
            };
        }
        let mut columns: Vec<Column> = Vec::with_capacity(self.cols.len());
        let mut value_only = BitSet::default();
        for (id, col) in self.cols.into_iter().enumerate() {
            if col.present {
                if col.any_bound {
                    // Promote homogeneous bound columns to the best stored
                    // representation so downstream operators can stay on
                    // typed vectors when the batch layout can represent them.
                    columns.push(classify_stored_column(col.values));
                } else {
                    // A slot that carried a value but was never bound in any
                    // row mirrors an env slot with a cleared bound bit (e.g.
                    // aggregate finalization). Record it so `is_bound_at` and
                    // the env reconstructions report it as unbound, and keep it
                    // as `Values` to preserve that distinction.
                    columns.push(Column::Values(col.values));
                    value_only.set(id);
                }
            } else {
                columns.push(Column::Unbound);
            }
        }
        let origin_rows = if self.any_origin {
            Some(self.origins)
        } else {
            None
        };
        Batch {
            len: self.rows,
            selection: None,
            columns,
            origin_rows,
            value_only,
            _marker: PhantomData,
        }
    }
}

/// A columnar batch of rows.
///
/// Each column corresponds to a variable slot
/// (by `Variable.id`). The `len` field indicates how many logical rows exist.
/// The optional `selection` vector enables zero-copy filtering.
#[derive(Clone)]
pub struct Batch<'a> {
    /// Number of logical rows in this batch (before selection filtering).
    len: usize,
    /// If `Some`, only these row indices are active (sorted, deduplicated).
    /// If `None`, all rows `0..len` are active.
    selection: Option<Vec<u16>>,
    /// One column per variable slot. Indexed by `Variable.id`.
    /// Used by native batch operators. Empty when `envs` is set.
    columns: Vec<Column>,
    /// Per-row correlation tag for columnar batches, indexed by logical row.
    /// When the batch is env-backed, the tag lives in each `Env::origin_row`
    /// instead; this sidecar is the columnar equivalent. `None` means every
    /// row's origin is `0` (the default for uncorrelated plans).
    origin_rows: Option<Vec<u32>>,
    /// Var ids whose `Column::Values` holds a value that is *present but not
    /// bound* (e.g. aggregate accumulator aliasing): `value_at` returns the
    /// value but `is_bound_at` reports `false`, and `BatchRow::to_owned_row`
    /// reconstructs the slot as unbound so downstream `Row::merge` skips it.
    value_only: BitSet,
    /// Retains the `'a` lifetime carried by the batch's borrowed data.
    _marker: PhantomData<&'a ()>,
}

impl<'a> Batch<'a> {
    /// Creates an empty batch with the given number of column slots.
    #[must_use]
    pub fn new(num_columns: usize) -> Self {
        let mut columns = Vec::with_capacity(num_columns);
        for _ in 0..num_columns {
            columns.push(Column::Unbound);
        }
        Self {
            len: 0,
            selection: None,
            columns,
            origin_rows: None,
            value_only: BitSet::default(),
            _marker: PhantomData,
        }
    }

    /// Creates a batch from fully materialized columns.
    #[must_use]
    pub fn from_columns(columns: impl IntoIterator<Item = Column>) -> Self {
        let mut batch = Self::new(0);
        for (i, col) in columns.into_iter().enumerate() {
            batch.set_column(i as u32, col);
        }
        batch
    }

    /// Compacts this batch in place by applying its selection vector, yielding
    /// a dense [`Batch`] whose logical rows are exactly the previously-active
    /// rows (in selection order) and whose `selection` is `None`. Typed columns
    /// are preserved (no collapse to [`Column::Values`]), so this is the purely
    /// columnar replacement for the old `clone_active_rows` round-trip: callers
    /// that own a batch move it through here instead of deep-cloning, and those
    /// that only borrow one pair `clone()` with `into_compacted()`.
    ///
    /// A batch with no selection is already dense and is returned unchanged.
    #[must_use]
    pub fn into_compacted(self) -> Self {
        let Some(sel) = self.selection.as_ref() else {
            return self;
        };
        let indices: Vec<usize> = sel.iter().map(|&i| i as usize).collect();
        let mut batch = self.gather(&indices);
        batch.selection = None;
        batch
    }

    /// Creates a new batch by gathering rows from this batch at the given
    /// indices. Indices may be repeated or out of order.
    #[must_use]
    pub fn gather(
        &self,
        indices: &[usize],
    ) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|c| c.gather(indices.iter().copied()))
            .collect();
        let origin_rows = self.origin_rows.as_ref().and_then(|o| {
            let origins: Vec<u32> = indices.iter().map(|&i| o[i]).collect();
            origins.iter().any(|&x| x != 0).then_some(origins)
        });
        Batch {
            len: indices.len(),
            selection: None,
            columns,
            origin_rows,
            value_only: self.value_only.clone(),
            _marker: PhantomData,
        }
    }

    /// Snapshots every active row into a fresh, dense columnar [`Batch`] and
    /// stamps each emitted row's `origin_row` with its sequential position
    /// (`0..n`) in active order. This is the columnar replacement for the
    /// correlated argument-batch idiom that clones active rows while setting
    /// `e.origin_row = i` (Apply / Optional / SemiApply / Merge / OR-apply),
    /// used downstream to correlate sub-plan results back to outer rows.
    ///
    /// The `origin_rows` sidecar is emitted only when it would contain a
    /// non-zero entry (i.e. `n > 1`).
    #[must_use]
    pub fn clone_active_rows_seq_origin(&self) -> Batch<'a> {
        let mut batch = self.clone().into_compacted();
        let n = batch.len;
        batch.origin_rows = (n > 1).then(|| (0..n as u32).collect());
        batch
    }

    /// Returns the number of logical rows in this batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of active (non-filtered) rows.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.selection.as_ref().map_or(self.len, Vec::len)
    }

    /// Returns true if there are no active rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active_len() == 0
    }

    /// Returns the selection vector, if any.
    #[must_use]
    pub fn selection(&self) -> Option<&[u16]> {
        self.selection.as_deref()
    }

    /// Sets the selection vector.
    pub fn set_selection(
        &mut self,
        sel: Vec<u16>,
    ) {
        self.selection = Some(sel);
    }

    /// Returns an iterator over active row indices.
    #[must_use]
    pub const fn active_indices(&self) -> ActiveIndices<'_, 'a> {
        ActiveIndices {
            batch: self,
            pos: 0,
        }
    }

    /// Returns a reference to the column at the given variable id.
    #[must_use]
    pub fn column(
        &self,
        var_id: u32,
    ) -> &Column {
        let idx = var_id as usize;
        if idx < self.columns.len() {
            &self.columns[idx]
        } else {
            &Column::Unbound
        }
    }

    /// Returns a mutable reference to the column at the given variable id.
    /// Grows the columns vector if needed.
    pub fn column_mut(
        &mut self,
        var_id: u32,
    ) -> &mut Column {
        let idx = var_id as usize;
        while self.columns.len() <= idx {
            self.columns.push(Column::Unbound);
        }
        &mut self.columns[idx]
    }

    /// Sets a column at the given variable id.
    pub fn set_column(
        &mut self,
        var_id: u32,
        col: Column,
    ) {
        let col = match col {
            // Upgrade value-backed columns to the best lossless stored shape.
            Column::Values(values) => classify_stored_column(values),
            other => other,
        };

        let col_len = col.len();
        if self.len == 0 {
            self.len = col_len;
        } else if !matches!(col, Column::Unbound) {
            debug_assert_eq!(
                col_len, self.len,
                "column length {} must match batch len {}",
                col_len, self.len
            );
        }

        let idx = var_id as usize;
        while self.columns.len() <= idx {
            self.columns.push(Column::Unbound);
        }
        self.columns[idx] = col;
        // Installing an explicit column binds the slot; clear any stale
        // value-only marker so the new binding is reported as bound.
        self.value_only.clear(idx);
    }

    /// Read a single value by (var_id, row) from the typed columns. Returns
    /// `None` when the variable is unbound in this row (out-of-range slot or
    /// `Column::Unbound`); returns `Some(Value::Null)` for an explicitly-null
    /// binding. Clones the value.
    #[must_use]
    pub fn value_at(
        &self,
        var_id: u32,
        row: usize,
    ) -> Option<Value> {
        match self.column(var_id) {
            Column::Unbound => None,
            col => Some(col.get(row)),
        }
    }

    /// Compare two rows of a single column for a sort tiebreaker, treating an
    /// unbound column as `Null` on both sides (and therefore equal). Unlike
    /// pairing two [`value_at`](Self::value_at) calls, this borrows stored
    /// heterogeneous `Value`s instead of cloning them — avoiding an allocation
    /// per comparison for `String`/`List`/`Map` columns, which can dominate the
    /// comparator when many rows share equal primary sort keys.
    #[must_use]
    pub fn compare_rows_at(
        &self,
        var_id: u32,
        a: usize,
        b: usize,
    ) -> Ordering {
        match self.column(var_id) {
            Column::Unbound => Ordering::Equal,
            Column::Values(vals) => vals[a].compare_value(&vals[b]).0,
            col => col.get(a).compare_value(&col.get(b)).0,
        }
    }

    /// Returns the per-row correlation tag for `row`. Defaults to `0` when no
    /// origin has been assigned (uncorrelated plans).
    #[must_use]
    pub fn origin_row(
        &self,
        row: usize,
    ) -> u32 {
        self.origin_rows.as_ref().map_or(0, |o| o[row])
    }

    /// Installs the columnar per-row correlation sidecar. The vector is indexed
    /// by logical row (length must equal [`len`](Self::len)). Ignored for
    /// env-backed batches, which carry the tag inside each `Env`.
    pub fn set_origin_rows(
        &mut self,
        origins: Vec<u32>,
    ) {
        debug_assert_eq!(origins.len(), self.len);
        self.origin_rows = Some(origins);
    }

    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the variable `var_id` is explicitly bound in `row`.
    /// Column-level: a non-`Unbound` column is considered bound for every row,
    /// unless the slot is tracked as value-present-but-unbound.
    #[must_use]
    pub fn is_bound_at(
        &self,
        var_id: u32,
        row: usize,
    ) -> bool {
        let _ = row;
        !matches!(self.column(var_id), Column::Unbound) && !self.value_only.test(var_id as usize)
    }

    /// Write an entire column of values into the active rows.
    /// `values.len()` must equal `self.active_len()`.
    pub fn write_column(
        &mut self,
        var_id: u32,
        values: Vec<Value>,
    ) {
        // Scatter the active-row values into a full-length column, preserving
        // any existing bindings on non-active rows.
        if let Some(sel) = self.selection.clone() {
            debug_assert_eq!(values.len(), sel.len());
            let mut full: Vec<Value> = (0..self.len)
                .map(|r| self.value_at(var_id, r).unwrap_or(Value::Null))
                .collect();
            for (val, &row) in values.into_iter().zip(sel.iter()) {
                full[row as usize] = val;
            }
            self.set_column(var_id, Column::Values(full));
        } else {
            debug_assert_eq!(values.len(), self.len);
            self.set_column(var_id, Column::Values(values));
        }
    }

    /// Takes a column out of this batch, replacing it with `Unbound`.
    pub fn take_column(
        &mut self,
        var_id: u32,
    ) -> Column {
        let idx = var_id as usize;
        if idx < self.columns.len() {
            self.value_only.clear(idx);
            std::mem::replace(&mut self.columns[idx], Column::Unbound)
        } else {
            Column::Unbound
        }
    }

    /// Extracts node IDs for a given variable from this batch.
    /// Returns `None` if the variable doesn't hold node values.
    #[must_use]
    pub fn extract_node_ids(
        &self,
        var_id: u32,
    ) -> Option<Vec<NodeId>> {
        match self.column(var_id) {
            Column::NodeIds(ids) => Some(ids.clone()),
            _ => None,
        }
    }
}

/// A borrowed view of a single row of a [`Batch`], implementing [`RowView`]
/// so the expression evaluator can read columnar data without materializing
/// an owned [`Row`].
pub struct BatchRow<'b, 'a> {
    batch: &'b Batch<'a>,
    row: usize,
}

impl<'b, 'a> BatchRow<'b, 'a> {
    /// Creates a view of `row` in `batch`.
    #[must_use]
    pub const fn new(
        batch: &'b Batch<'a>,
        row: usize,
    ) -> Self {
        Self { batch, row }
    }
}

impl RowView for BatchRow<'_, '_> {
    fn value_at(
        &self,
        var_id: u32,
    ) -> Option<Value> {
        // Honor the `RowView` contract: an in-range slot that is unbound reads
        // back as `Some(Null)` (matching the owned `Row` and the legacy
        // env-backed runtime), while a slot beyond the batch's column space is
        // genuinely out of scope and reads back as `None`.
        match self.batch.value_at(var_id, self.row) {
            Some(value) => Some(value),
            None if (var_id as usize) < self.batch.num_columns() => Some(Value::Null),
            None => None,
        }
    }

    fn to_owned_row(&self) -> Row {
        let mut r = Row::new();
        for (var_id, col) in self.batch.columns.iter().enumerate() {
            if !matches!(col, Column::Unbound) {
                r.insert_by_id(var_id as u32, col.get(self.row));
                if self.batch.value_only.test(var_id) {
                    r.unbind_by_id(var_id as u32);
                }
            }
        }
        if let Some(origins) = &self.batch.origin_rows {
            r.origin_row = origins[self.row];
        }
        r
    }
}

/// Iterator over active row indices in a batch.
pub struct ActiveIndices<'b, 'a> {
    batch: &'b Batch<'a>,
    pos: usize,
}

impl Iterator for ActiveIndices<'_, '_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.batch.selection {
            Some(sel) => {
                if self.pos < sel.len() {
                    let idx = sel[self.pos] as usize;
                    self.pos += 1;
                    Some(idx)
                } else {
                    None
                }
            }
            None => {
                if self.pos < self.batch.len {
                    let idx = self.pos;
                    self.pos += 1;
                    Some(idx)
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .batch
            .selection
            .as_ref()
            .map_or_else(|| self.batch.len - self.pos, |sel| sel.len() - self.pos);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ActiveIndices<'_, '_> {}

// ---------------------------------------------------------------------------
// BatchOp — enum dispatch for batch-mode operators
// ---------------------------------------------------------------------------

/// Batch-mode operator enum. Each variant wraps a concrete operator that
/// processes data in batches of up to [`BATCH_SIZE`] rows.
#[allow(clippy::large_enum_variant)]
pub enum BatchOp<'a> {
    /// Yields a single batch containing one default Env row. Used as the
    /// leaf of operator trees when no child exists (e.g. `RETURN 1`).
    Once(Option<Batch<'a>>),
    /// Argument leaf for correlated sub-plans. Receives a batch via
    /// `set_argument_batch` and yields it.
    Argument(Option<Batch<'a>>),
    /// Scan nodes by label.
    NodeByLabelScan(NodeByLabelScanOp<'a>),
    /// Augments child scan with pending-created nodes and filters deleted/removed.
    IncludePending(IncludePendingOp<'a>),
    /// Filter rows by predicate.
    Filter(FilterOp<'a>),
    /// Project expressions into new columns.
    Project(ProjectOp<'a>),
    /// Skip first N rows.
    Skip(SkipOp<'a>),
    /// Limit output to N rows.
    Limit(LimitOp<'a>),
    /// Remove duplicate rows.
    Distinct(DistinctOp<'a>),
    /// Sort rows by expressions.
    Sort(SortOp<'a>),
    /// Aggregate rows by keys.
    Aggregate(AggregateOp<'a>),
    /// Unwind lists into rows.
    Unwind(UnwindOp<'a>),
    /// Conditional traverse relationships.
    CondTraverse(CondTraverseOp<'a>),
    /// Expand into existing relationships.
    ExpandInto(ExpandIntoOp<'a>),
    /// Seek nodes by internal ID.
    NodeByIdSeek(NodeByIdSeekOp<'a>),
    /// Scan nodes by index.
    NodeByIndexScan(NodeByIndexScanOp<'a>),
    /// Scan edges by index.
    EdgeByIndexScan(EdgeByIndexScanOp<'a>),
    /// Cartesian product of sub-plans.
    CartesianProduct(CartesianProductOp<'a>),
    /// Correlated sub-query execution.
    Apply(ApplyOp<'a>),
    /// Existence-based filtering via sub-plan.
    SemiApply(SemiApplyOp<'a>),
    /// Optional match with NULL fallback.
    Optional(OptionalOp<'a>),
    /// Create nodes/relationships.
    Create(CreateOp<'a>),
    /// Delete nodes/relationships.
    Delete(DeleteOp<'a>),
    /// Set properties.
    Set(SetOp<'a>),
    /// Remove properties/labels.
    Remove(RemoveOp<'a>),
    /// Match-or-create.
    Merge(MergeOp<'a>),
    /// Commit pending mutations.
    Commit(CommitOp<'a>),
    /// Concatenate results from multiple sub-plans.
    Union(UnionOp<'a>),
    /// Build path values.
    PathBuilder(PathBuilderOp<'a>),
    /// Load data from CSV files.
    LoadCsv(LoadCsvOp<'a>),
    /// Call stored procedures.
    ProcedureCall(ProcedureCallOp<'a>),
    /// Fulltext index scan.
    NodeByFulltextScan(NodeByFulltextScanOp<'a>),
    /// Edge fulltext index scan.
    EdgeByFulltextScan(EdgeByFulltextScanOp<'a>),
    /// KNN vector index scan over node labels.
    NodeByVectorScan(NodeByVectorScanOp<'a>),
    /// KNN vector index scan over relationship types.
    EdgeByVectorScan(EdgeByVectorScanOp<'a>),
    /// Combined label + ID scan.
    NodeByLabelAndIdScan(NodeByLabelAndIdScanOp<'a>),
    /// Variable-length relationship traverse.
    CondVarLenTraverse(CondVarLenTraverseOp<'a>),
    AllShortestPaths(AllShortestPathsOp<'a>),
    /// OR-apply multiplexer for disjunctive patterns.
    OrApplyMultiplexer(OrApplyMultiplexerOp<'a>),
    /// FOREACH loop operator.
    ForEach(ForEachOp<'a>),
    /// Value Hash Join: hash-based equi-join of two sub-plans.
    ValueHashJoin(ValueHashJoinOp<'a>),
}

impl<'a> BatchOp<'a> {
    /// Propagates a batch down to `Argument` leaves in the operator tree.
    /// Each operator delegates to its child(ren) until an `Argument` leaf
    /// is reached, where the batch is installed.
    pub fn set_argument_batch(
        &mut self,
        batch: Batch<'a>,
    ) {
        match self {
            Self::Argument(slot) => {
                *slot = Some(batch);
            }
            Self::Once(_) => {}
            Self::ProcedureCall(op) => {
                op.reset_state();
                op.child.set_argument_batch(batch);
            }
            Self::NodeByLabelScan(op) => op.child.set_argument_batch(batch),
            Self::IncludePending(op) => {
                op.capture_argument(&batch);
                op.child.set_argument_batch(batch);
            }
            Self::Filter(op) => op.child.set_argument_batch(batch),
            Self::Project(op) => op.child.set_argument_batch(batch),
            Self::Skip(op) => op.child.set_argument_batch(batch),
            Self::Limit(op) => op.child.set_argument_batch(batch),
            Self::Distinct(op) => op.child.set_argument_batch(batch),
            Self::Sort(op) => {
                if let Some(ref mut c) = op.child {
                    c.set_argument_batch(batch);
                }
            }
            Self::Aggregate(op) => {
                if let Some(ref mut c) = op.child {
                    c.set_argument_batch(batch);
                }
            }
            Self::Unwind(op) => op.child.set_argument_batch(batch),
            Self::CondTraverse(op) => op.child.set_argument_batch(batch),
            Self::ExpandInto(op) => op.child.set_argument_batch(batch),
            Self::NodeByIdSeek(op) => op.child.set_argument_batch(batch),
            Self::NodeByIndexScan(op) => op.child.set_argument_batch(batch),
            Self::EdgeByIndexScan(op) => op.child.set_argument_batch(batch),
            Self::CartesianProduct(op) => {
                for right_child in &mut op.right_children {
                    right_child.set_argument_batch(batch.clone().into_compacted());
                }
                op.child.set_argument_batch(batch);
            }
            Self::Apply(op) => op.child.set_argument_batch(batch),
            Self::SemiApply(op) => op.child.set_argument_batch(batch),
            Self::Optional(op) => op.child.set_argument_batch(batch),
            Self::Create(op) => op.child.set_argument_batch(batch),
            Self::Delete(op) => op.child.set_argument_batch(batch),
            Self::Set(op) => op.child.set_argument_batch(batch),
            Self::Remove(op) => op.child.set_argument_batch(batch),
            Self::Merge(op) => op.child.set_argument_batch(batch),
            Self::Commit(op) => {
                if let Some(ref mut c) = op.child {
                    c.set_argument_batch(batch);
                }
            }
            Self::Union(op) => {
                op.store_argument_batch(batch);
                if let Some(ref mut c) = op.current
                    && let Some(ref arg) = op.argument_batch
                {
                    c.set_argument_batch(arg.clone());
                }
            }
            Self::PathBuilder(op) => op.child.set_argument_batch(batch),
            Self::LoadCsv(op) => op.child.set_argument_batch(batch),
            Self::NodeByFulltextScan(op) => op.child.set_argument_batch(batch),
            Self::EdgeByFulltextScan(op) => op.child.set_argument_batch(batch),
            Self::NodeByVectorScan(op) => {
                // Drop any KNN rows still queued from the previous
                // outer iteration; otherwise correlated plans (Apply)
                // can leak rows across outer batches when the inner
                // side stops early.
                op.emitter.reset();
                op.child.set_argument_batch(batch);
            }
            Self::EdgeByVectorScan(op) => {
                op.emitter.reset();
                op.child.set_argument_batch(batch);
            }
            Self::NodeByLabelAndIdScan(op) => op.child.set_argument_batch(batch),
            Self::CondVarLenTraverse(op) => op.child.set_argument_batch(batch),
            Self::AllShortestPaths(op) => op.child.set_argument_batch(batch),
            Self::OrApplyMultiplexer(op) => op.child.set_argument_batch(batch),
            Self::ForEach(op) => op.child.set_argument_batch(batch),
            Self::ValueHashJoin(op) => {
                // Clear cached state so the join rematerializes for the new batch
                op.hash_table = None;
                op.right_batches.clear();
                op.left_batch = None;
                op.left_pos = 0;
                op.right_match_envs.clear();
                op.right_match_pos = 0;
                op.right.set_argument_batch(batch.clone().into_compacted());
                op.child.set_argument_batch(batch);
            }
        }
    }

    /// Returns the `(runtime, idx)` pair for this operator, used for
    /// inspect/record support. Returns `None` for synthetic leaves
    /// (`Once`, `Argument`) which have no associated IR node.
    const fn inspect_context(&self) -> Option<(&Runtime<'a>, NodeIdx<Dyn<IR>>)> {
        match self {
            Self::Once(_) | Self::Argument(_) => None,
            Self::NodeByLabelScan(op) => Some((op.runtime, op.idx)),
            Self::IncludePending(op) => Some((op.runtime, op.idx)),
            Self::Filter(op) => Some((op.runtime, op.idx)),
            Self::Project(op) => Some((op.runtime, op.idx)),
            Self::Skip(op) => Some((op.runtime, op.idx)),
            Self::Limit(op) => Some((op.runtime, op.idx)),
            Self::Distinct(op) => Some((op.runtime, op.idx)),
            Self::Sort(op) => Some((op.runtime, op.idx)),
            Self::Aggregate(op) => Some((op.runtime, op.idx)),
            Self::Unwind(op) => Some((op.runtime, op.idx)),
            Self::CondTraverse(op) => Some((op.runtime, op.idx)),
            Self::ExpandInto(op) => Some((op.runtime, op.idx)),
            Self::NodeByIdSeek(op) => Some((op.runtime, op.idx)),
            Self::NodeByIndexScan(op) => Some((op.runtime, op.idx)),
            Self::EdgeByIndexScan(op) => Some((op.runtime, op.idx)),
            Self::CartesianProduct(op) => Some((op.runtime, op.idx)),
            Self::Apply(op) => Some((op.runtime, op.idx)),
            Self::SemiApply(op) => Some((op.runtime, op.idx)),
            Self::Optional(op) => Some((op.runtime, op.idx)),
            Self::Create(op) => Some((op.runtime, op.idx)),
            Self::Delete(op) => Some((op.runtime, op.idx)),
            Self::Set(op) => Some((op.runtime, op.idx)),
            Self::Remove(op) => Some((op.runtime, op.idx)),
            Self::Merge(op) => Some((op.runtime, op.idx)),
            Self::Commit(op) => Some((op.runtime, op.idx)),
            Self::Union(op) => Some((op.runtime, op.idx)),
            Self::PathBuilder(op) => Some((op.runtime, op.idx)),
            Self::LoadCsv(op) => Some((op.runtime, op.idx)),
            Self::ProcedureCall(op) => Some((op.runtime, op.idx)),
            Self::NodeByFulltextScan(op) => Some((op.runtime, op.idx)),
            Self::EdgeByFulltextScan(op) => Some((op.runtime, op.idx)),
            Self::NodeByVectorScan(op) => Some((op.runtime, op.idx)),
            Self::EdgeByVectorScan(op) => Some((op.runtime, op.idx)),
            Self::NodeByLabelAndIdScan(op) => Some((op.runtime, op.idx)),
            Self::CondVarLenTraverse(op) => Some((op.runtime, op.idx)),
            Self::AllShortestPaths(op) => Some((op.runtime, op.idx)),
            Self::OrApplyMultiplexer(op) => Some((op.runtime, op.idx)),
            Self::ForEach(op) => Some((op.runtime, op.idx)),
            Self::ValueHashJoin(op) => Some((op.runtime, op.idx)),
        }
    }
}

impl<'a> Iterator for BatchOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check timeout and memory capacity before dispatching to the next operator.
        // Also capture profiling state. We must not hold a reference to `self`
        // across the dispatch, so everything is extracted up front.
        let profiling = if let Some((runtime, idx)) = self.inspect_context() {
            if let Err(e) = runtime.check_timeout() {
                return Some(Err(e));
            }
            if let Err(e) = runtime.check_mem_capacity() {
                return Some(Err(e));
            }
            if runtime.profile {
                let saved = runtime.profile_child_time.get();
                runtime.profile_child_time.set(std::time::Duration::ZERO);
                Some((idx, saved, std::time::Instant::now()))
            } else {
                None
            }
        } else {
            None
        };

        let result = match self {
            Self::Once(batch) | Self::Argument(batch) => batch.take().map(Ok),
            Self::NodeByLabelScan(op) => op.next(),
            Self::IncludePending(op) => op.next(),
            Self::Filter(op) => op.next(),
            Self::Project(op) => op.next(),
            Self::Skip(op) => op.next(),
            Self::Limit(op) => op.next(),
            Self::Distinct(op) => op.next(),
            Self::Sort(op) => op.next(),
            Self::Aggregate(op) => op.next(),
            Self::Unwind(op) => op.next(),
            Self::CondTraverse(op) => op.next(),
            Self::ExpandInto(op) => op.next(),
            Self::NodeByIdSeek(op) => op.next(),
            Self::NodeByIndexScan(op) => op.next(),
            Self::EdgeByIndexScan(op) => op.next(),
            Self::CartesianProduct(op) => op.next(),
            Self::Apply(op) => op.next(),
            Self::SemiApply(op) => op.next(),
            Self::Optional(op) => op.next(),
            Self::Create(op) => op.next(),
            Self::Delete(op) => op.next(),
            Self::Set(op) => op.next(),
            Self::Remove(op) => op.next(),
            Self::Merge(op) => op.next(),
            Self::Commit(op) => op.next(),
            Self::Union(op) => op.next(),
            Self::PathBuilder(op) => op.next(),
            Self::LoadCsv(op) => op.next(),
            Self::ProcedureCall(op) => op.next(),
            Self::NodeByFulltextScan(op) => op.next(),
            Self::EdgeByFulltextScan(op) => op.next(),
            Self::NodeByVectorScan(op) => op.next(),
            Self::EdgeByVectorScan(op) => op.next(),
            Self::NodeByLabelAndIdScan(op) => op.next(),
            Self::CondVarLenTraverse(op) => op.next(),
            Self::AllShortestPaths(op) => op.next(),
            Self::OrApplyMultiplexer(op) => op.next(),
            Self::ForEach(op) => op.next(),
            Self::ValueHashJoin(op) => op.next(),
        };

        if let Some(ref res) = result
            && let Some((runtime, idx)) = self.inspect_context()
        {
            // Record profiling data after dispatch.
            if let Some((prof_idx, saved_child_time, start)) = profiling {
                debug_assert_eq!(idx, prof_idx);
                let inclusive = start.elapsed();
                let child_time = runtime.profile_child_time.get();
                let self_time = inclusive.saturating_sub(child_time);
                let records = res.as_ref().map_or(0, Batch::active_len);
                let mut pd = runtime.profile_data.borrow_mut();
                let entry = pd.entry(idx).or_insert((0, std::time::Duration::ZERO));
                entry.0 += records;
                entry.1 += self_time;
                runtime.profile_child_time.set(saved_child_time + inclusive);
            }
            runtime.inspect_batch(idx, res);
        } else if let Some((prof_idx, saved_child_time, start)) = profiling {
            // Result is None (iterator exhausted) — still need to restore child time.
            if let Some((runtime, _)) = self.inspect_context() {
                let inclusive = start.elapsed();
                let child_time = runtime.profile_child_time.get();
                let self_time = inclusive.saturating_sub(child_time);
                let mut pd = runtime.profile_data.borrow_mut();
                let entry = pd.entry(prof_idx).or_insert((0, std::time::Duration::ZERO));
                entry.1 += self_time;
                runtime.profile_child_time.set(saved_child_time + inclusive);
            }
        }
        result
    }
}
