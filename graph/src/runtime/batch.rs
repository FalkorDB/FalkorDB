//! Columnar batch representation for vectorized query execution.
//!
//! A [`Batch`] stores multiple rows (up to [`BATCH_SIZE`] = 1024), enabling
//! operators to amortize per-row dispatch overhead and exploit data locality.
//!
//! ```text
//!  Batch (env-backed mode)
//! ┌────────────────────────────────────────────────────────┐
//! │  len: 4                                                │
//! │  selection: Some([0, 2, 3])   ← only these rows active │
//! │                                                        │
//! │  envs: [ Env0, Env1, Env2, Env3 ]                     │
//! │         ^^^^         ^^^^  ^^^^                         │
//! │         active       active active                     │
//! └────────────────────────────────────────────────────────┘
//!
//!  Batch (columnar mode, future)
//! ┌────────────────────────────────────────────────────────┐
//! │  columns[0]: NodeIds  [n1, n2, n3, n4]                 │
//! │  columns[1]: Ints     [10, 20, 30, 40]   ← SIMD ops   │
//! │  columns[2]: Values   ["a","b","c","d"]                │
//! │  selection:  None     ← all rows active                │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Dual Storage Model
//!
//! The batch supports two storage modes:
//! - **Env-backed** (`envs` field): row-oriented via `Vec<Env>`. Used by most
//!   operators today. Access via `env_ref()`, `get()`, `set()`, `read_columns()`.
//! - **Columnar** (`columns` field): typed columns for vectorized kernels.
//!   Used by `FilterOp`/`ProjectOp` for bulk property comparison.
//!
//! ## Zero-Copy Filtering
//!
//! Instead of removing filtered-out rows, operators set a **selection vector**
//! (`Vec<u16>`) listing active row indices. Downstream operators iterate only
//! the active rows via `active_indices()` / `active_env_iter()`.

use crate::graph::graph::{NodeId, RelationshipId};
use crate::planner::IR;
use crate::runtime::env::Env;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use orx_tree::{Dyn, NodeIdx};

use super::ops::aggregate::AggregateOp;
use super::ops::all_shortest_paths::AllShortestPathsOp;
use super::ops::apply::ApplyOp;
use super::ops::cartesian_product::CartesianProductOp;
use super::ops::commit::CommitOp;
use super::ops::cond_traverse::CondTraverseOp;
use super::ops::cond_var_len_traverse::CondVarLenTraverseOp;
use super::ops::create::CreateOp;
use super::ops::delete::DeleteOp;
use super::ops::edge_by_index_scan::EdgeByIndexScanOp;
use super::ops::distinct::DistinctOp;
use super::ops::expand_into::ExpandIntoOp;
use super::ops::filter::FilterOp;
use super::ops::foreach::ForEachOp;
use super::ops::limit::LimitOp;
use super::ops::load_csv::LoadCsvOp;
use super::ops::merge::MergeOp;
use super::ops::node_by_fulltext_scan::NodeByFulltextScanOp;
use super::ops::node_by_id_seek::NodeByIdSeekOp;
use super::ops::node_by_index_scan::NodeByIndexScanOp;
use super::ops::node_by_label_and_id_scan::NodeByLabelAndIdScanOp;
use super::ops::node_by_label_scan::NodeByLabelScanOp;
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

/// Maximum number of rows in a single batch.
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
pub enum Column {
    /// All values are node IDs (from scan/traverse operators).
    NodeIds(Vec<NodeId>),
    /// All values are relationship triples: (rel_id, src_node, dst_node).
    RelTriples(Vec<(RelationshipId, NodeId, NodeId)>),
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
            Self::RelTriples(triples) => {
                let (rel_id, src, dst) = triples[row];
                Value::Relationship(Box::new((rel_id, src, dst)))
            }
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
            Self::RelTriples(v) => v.len(),
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
}

/// A columnar batch of rows.
///
/// Each column corresponds to a variable slot
/// (by `Variable.id`). The `len` field indicates how many logical rows exist.
/// The optional `selection` vector enables zero-copy filtering.
///
/// When created via `from_envs`, the `envs` field stores the original `Env`
/// objects directly for lossless round-tripping.
pub struct Batch<'a> {
    /// Number of logical rows in this batch (before selection filtering).
    len: usize,
    /// If `Some`, only these row indices are active (sorted, deduplicated).
    /// If `None`, all rows `0..len` are active.
    selection: Option<Vec<u16>>,
    /// One column per variable slot. Indexed by `Variable.id`.
    /// Used by native batch operators. Empty when `envs` is set.
    columns: Vec<Column>,
    /// Raw env storage for the adapter path. When set, `row_to_env`
    /// returns a clone from here instead of reconstructing from columns.
    envs: Option<Vec<Env<'a>>>,
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
            envs: None,
        }
    }

    /// Creates a batch from a vector of `Env` rows.
    #[must_use]
    pub const fn from_envs(envs: Vec<Env<'a>>) -> Self {
        let len = envs.len();
        Self {
            len,
            selection: None,
            columns: Vec::new(),
            envs: Some(envs),
        }
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

    /// Returns an iterator yielding a reference to the [`Env`] for each active row.
    #[must_use]
    pub const fn active_env_iter<'b>(&'b self) -> ActiveEnvIter<'b, 'a> {
        ActiveEnvIter {
            batch: self,
            indices: self.active_indices(),
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
        let idx = var_id as usize;
        while self.columns.len() <= idx {
            self.columns.push(Column::Unbound);
        }
        self.columns[idx] = col;
    }

    /// Read a single value by (row, var_id).
    /// Returns `&Value::Null` for unbound slots. No allocation.
    #[must_use]
    pub fn get(
        &self,
        row: usize,
        var_id: u32,
    ) -> &Value {
        static NULL: Value = Value::Null;
        let envs = self.envs.as_ref().expect("batch must be env-backed");
        envs[row].get_by_id(var_id).unwrap_or(&NULL)
    }

    /// Write a single value by (row, var_id). Mutates the env in-place.
    pub fn set(
        &mut self,
        row: usize,
        var_id: u32,
        value: Value,
    ) {
        let envs = self.envs.as_mut().expect("batch must be env-backed");
        envs[row].insert_by_id(var_id, value);
    }

    /// Read multiple variables for all active rows, returned in row-major order.
    /// Outer index = active row (length `self.active_len()`), inner index =
    /// variable (same order as `var_ids`).
    ///
    /// Env storage is already row-major, so this is the natural access
    /// pattern — no transpose needed.
    #[must_use]
    pub fn read_columns(
        &self,
        var_ids: &[u32],
    ) -> Vec<Vec<&Value>> {
        static NULL: Value = Value::Null;
        let envs = self.envs.as_ref().expect("batch must be env-backed");
        self.active_indices()
            .map(|row| {
                var_ids
                    .iter()
                    .map(|&id| envs[row].get_by_id(id).unwrap_or(&NULL))
                    .collect()
            })
            .collect()
    }

    /// Returns a mutable slice of the underlying env storage.
    /// Panics if the batch is not env-backed.
    pub const fn envs_mut(&mut self) -> &mut [Env<'a>] {
        self.envs
            .as_mut()
            .expect("batch must be env-backed")
            .as_mut_slice()
    }

    /// Returns a shared reference to the Env at the given row index.
    /// Panics if the batch is not env-backed.
    /// This is zero-copy — no allocation or cloning.
    #[must_use]
    pub fn env_ref(
        &self,
        row: usize,
    ) -> &Env<'a> {
        let envs = self.envs.as_ref().expect("batch must be env-backed");
        &envs[row]
    }

    /// Write an entire column of values into the active rows.
    /// `values.len()` must equal `self.active_len()`.
    pub fn write_column(
        &mut self,
        var_id: u32,
        values: Vec<Value>,
    ) {
        let envs = self.envs.as_mut().expect("batch must be env-backed");
        if let Some(sel) = &self.selection {
            debug_assert_eq!(values.len(), sel.len());
            for (val, &row) in values.into_iter().zip(sel.iter()) {
                envs[row as usize].insert_by_id(var_id, val);
            }
        } else {
            debug_assert_eq!(values.len(), self.len);
            for (row, val) in values.into_iter().enumerate() {
                envs[row].insert_by_id(var_id, val);
            }
        }
    }

    /// Consumes the batch and returns the underlying env storage.
    /// Panics if the batch is not env-backed.
    #[must_use]
    pub fn into_envs(self) -> Vec<Env<'a>> {
        self.envs.expect("batch must be env-backed")
    }

    /// Takes a column out of this batch, replacing it with `Unbound`.
    pub fn take_column(
        &mut self,
        var_id: u32,
    ) -> Column {
        let idx = var_id as usize;
        if idx < self.columns.len() {
            std::mem::replace(&mut self.columns[idx], Column::Unbound)
        } else {
            Column::Unbound
        }
    }

    /// Extracts node IDs for a given variable from this batch.
    /// Works for both env-backed and columnar batches.
    /// Returns `None` if the variable doesn't hold node values.
    #[must_use]
    pub fn extract_node_ids(
        &self,
        var_id: u32,
    ) -> Option<Vec<NodeId>> {
        if let Some(envs) = &self.envs {
            let mut ids = Vec::with_capacity(self.len);
            for env in envs {
                match env.get_by_id(var_id)? {
                    Value::Node(id) => ids.push(*id),
                    _ => return None,
                }
            }
            Some(ids)
        } else {
            match self.column(var_id) {
                Column::NodeIds(ids) => Some(ids.clone()),
                _ => None,
            }
        }
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
// ActiveEnvIter — iterator yielding Env for each active row
// ---------------------------------------------------------------------------

/// Iterator over active rows in a batch, yielding a reference to the [`Env`] per row.
pub struct ActiveEnvIter<'b, 'a> {
    batch: &'b Batch<'a>,
    indices: ActiveIndices<'b, 'a>,
}

impl<'b, 'a> Iterator for ActiveEnvIter<'b, 'a> {
    type Item = &'b Env<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.indices.next()?;
        Some(self.batch.env_ref(idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl ExactSizeIterator for ActiveEnvIter<'_, '_> {}

// ---------------------------------------------------------------------------
// BatchOp — enum dispatch for batch-mode operators
// ---------------------------------------------------------------------------

/// Batch-mode operator enum. Each variant wraps a concrete operator that
/// processes data in batches of up to [`BATCH_SIZE`] rows.
pub enum BatchOp<'a> {
    /// Yields a single batch containing one default Env row. Used as the
    /// leaf of operator trees when no child exists (e.g. `RETURN 1`).
    Once(Option<Batch<'a>>),
    /// Argument leaf for correlated sub-plans. Receives a batch via
    /// `set_argument_batch` and yields it.
    Argument(Option<Batch<'a>>),
    /// Scan nodes by label.
    NodeByLabelScan(NodeByLabelScanOp<'a>),
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
                op.batches = None;
                op.child.set_argument_batch(batch);
            }
            Self::NodeByLabelScan(op) => op.child.set_argument_batch(batch),
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
                    let cloned: Vec<Env<'a>> = batch
                        .active_env_iter()
                        .map(|e| e.clone_pooled(op.runtime.env_pool))
                        .collect();
                    right_child.set_argument_batch(Batch::from_envs(cloned));
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
                op.store_argument_batch(&batch);
                if let Some(ref mut c) = op.current
                    && let Some(ref envs) = op.argument_batch
                {
                    let cloned: Vec<crate::runtime::env::Env<'a>> = envs
                        .iter()
                        .map(|e| e.clone_pooled(op.runtime.env_pool))
                        .collect();
                    c.set_argument_batch(Batch::from_envs(cloned));
                }
            }
            Self::PathBuilder(op) => op.child.set_argument_batch(batch),
            Self::LoadCsv(op) => op.child.set_argument_batch(batch),
            Self::NodeByFulltextScan(op) => op.child.set_argument_batch(batch),
            Self::NodeByLabelAndIdScan(op) => op.child.set_argument_batch(batch),
            Self::CondVarLenTraverse(op) => op.child.set_argument_batch(batch),
            Self::AllShortestPaths(op) => op.child.set_argument_batch(batch),
            Self::OrApplyMultiplexer(op) => op.child.set_argument_batch(batch),
            Self::ForEach(op) => op.child.set_argument_batch(batch),
            Self::ValueHashJoin(op) => {
                // Clear cached state so the join rematerializes for the new batch
                op.hash_table = None;
                op.left_envs.clear();
                op.left_pos = 0;
                op.right_match_envs.clear();
                op.right_match_pos = 0;
                let cloned: Vec<Env<'a>> = batch
                    .active_env_iter()
                    .map(|e| e.clone_pooled(op.runtime.env_pool))
                    .collect();
                op.right.set_argument_batch(Batch::from_envs(cloned));
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
        // Check if profiling is enabled and save state before dispatch.
        // We must not hold a reference to `self` across the dispatch.
        let profiling = self.inspect_context().and_then(|(runtime, idx)| {
            if runtime.profile {
                let saved = runtime.profile_child_time.get();
                runtime.profile_child_time.set(std::time::Duration::ZERO);
                Some((idx, saved, std::time::Instant::now()))
            } else {
                None
            }
        });

        let result = match self {
            Self::Once(batch) | Self::Argument(batch) => batch.take().map(Ok),
            Self::NodeByLabelScan(op) => op.next(),
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
