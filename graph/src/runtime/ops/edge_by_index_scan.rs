//! Batch-mode edge index scan operator — retrieves edges using a secondary index.
//!
//! For each active row in the input batch, evaluates the index query
//! parameters and collects matching edges. Supports equality, range,
//! and conjunctive (AND/OR) index queries.
//!
//! This operator replaces `CondTraverse` when the optimizer detects that
//! a filter on an edge property can be pushed into an edge index.
//!
//! ```text
//!  Input row ──► evaluate index query params
//!                      │
//!             ┌────────▼────────┐
//!             │ IndexQuery::    │
//!             │  Equal          │  key = value
//!             │  Range          │  min <= key <= max
//!             │  And([...])     │  conjunction of the above
//!             └────────┬────────┘
//!                      │
//!            graph.get_indexed_edges()
//!                      │
//!             output rows with edge + endpoint IDs
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::graph::{NodeId, RelationshipId};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::unsupported_by_native_index;
use crate::index::indexer::IndexQuery;
use crate::parser::ast::{ExprIR, QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    row::RowView,
    runtime::Runtime,
    value::Value,
};
// `NodeRef` supplies the `.root()` / `.idx()` methods used below via
// trait import — it appears unused in signatures but removing it
// breaks `value.root().idx()` etc.
use orx_tree::{Bfs, Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, EdgeEndpoints, RowIter};

type EdgeHit = (NodeId, NodeId, RelationshipId);

/// Walks hits shared through an `Arc`, by value and in order. The iterator owns its
/// `Arc`, so a row's iterator outlives the borrow of the cache that handed it out
/// and never copies the hits. Takes a kept group (`Arc<[EdgeHit]>`) or the type's
/// edge list (`Arc<Vec<EdgeHit>>`, kept as `get_all_edges` returns it, uncopied).
fn shared_hits<H>(hits: Arc<H>) -> impl Iterator<Item = EdgeHit>
where
    H: AsRef<[EdgeHit]> + ?Sized,
{
    let len = (*hits).as_ref().len();
    (0..len).map(move |i| (*hits).as_ref()[i])
}

/// The pattern's `(from, to)` nodes of a hit. Hits come in graph order
/// `(src, dst, edge)`; a transposed scan matches the pattern's `from` on the edge's
/// destination.
const fn pattern_endpoints(
    (src, dst, _): EdgeHit,
    transposed: bool,
) -> (NodeId, NodeId) {
    if transposed { (dst, src) } else { (src, dst) }
}

/// Which hits an input row keeps: those whose pattern endpoints match the
/// endpoints the row already binds, and only loops for a self-loop pattern like
/// `MATCH (n)-[r:T]->(n)` (it binds its one alias once, via `from`, so a non-loop
/// edge would leak through).
#[derive(Clone, Copy)]
struct EndpointFilter {
    bound_from: Option<NodeId>,
    bound_to: Option<NodeId>,
    same_endpoint_alias: bool,
    transposed: bool,
}

impl EndpointFilter {
    /// True when the filter keeps every hit: no endpoint is bound and the
    /// pattern is not a self-loop.
    const fn keeps_all(&self) -> bool {
        self.bound_from.is_none() && self.bound_to.is_none() && !self.same_endpoint_alias
    }

    fn keeps(
        &self,
        hit: EdgeHit,
    ) -> bool {
        let (from, to) = pattern_endpoints(hit, self.transposed);
        self.bound_from.is_none_or(|id| id == from)
            && self.bound_to.is_none_or(|id| id == to)
            && (!self.same_endpoint_alias || from == to)
    }
}

/// Every hit of one index query, oriented as the pattern matches them (an
/// undirected pattern lists an edge in both orientations), with by-endpoint
/// maps built on first use. Keys are the pattern's `from` / `to` node ids.
struct IndexedHits {
    all: Vec<EdgeHit>,
    by_from: Option<HashMap<NodeId, Arc<[EdgeHit]>>>,
    by_to: Option<HashMap<NodeId, Arc<[EdgeHit]>>>,
    /// What an endpoint with no hits gets: one empty slice per cache. Not
    /// `Arc::default()`, which is a single process-wide static, so every reader
    /// thread would count references on the same cache line.
    no_hits: Arc<[EdgeHit]>,
}

/// `hits` grouped by their pattern `from` node (`by_from`) or `to` node, each group
/// in list order.
fn group_by_endpoint(
    hits: &[EdgeHit],
    transposed: bool,
    by_from: bool,
) -> HashMap<NodeId, Arc<[EdgeHit]>> {
    let mut groups: HashMap<NodeId, Vec<EdgeHit>> = HashMap::new();
    for &hit in hits {
        let (from, to) = pattern_endpoints(hit, transposed);
        groups
            .entry(if by_from { from } else { to })
            .or_default()
            .push(hit);
    }
    groups
        .into_iter()
        .map(|(node, group)| (node, Arc::from(group)))
        .collect()
}

/// True when evaluating `expr` can not depend on the input row or change
/// between rows: it holds only literals, parameters and list/map constructors.
fn expr_is_row_invariant(expr: &QueryExpr<Variable>) -> bool {
    expr.root().indices::<Bfs>().all(|idx| {
        matches!(
            expr.node(idx).data(),
            ExprIR::Constant(_) | ExprIR::Parameter(_) | ExprIR::List | ExprIR::Map
        )
    })
}

fn query_is_row_invariant(q: &IndexQuery<QueryExpr<Variable>>) -> bool {
    match q {
        IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => {
            expr_is_row_invariant(value)
        }
        IndexQuery::Range { min, max, .. } => {
            min.iter().chain(max.iter()).all(expr_is_row_invariant)
        }
        IndexQuery::And(qs) | IndexQuery::Or(qs) => qs.iter().all(query_is_row_invariant),
        IndexQuery::Point { point, radius, .. } => {
            expr_is_row_invariant(point) && expr_is_row_invariant(radius)
        }
        IndexQuery::InList { list, .. } => expr_is_row_invariant(list),
    }
}

/// Every edge of `relationship_type`, `(src, dst, edge)`, read once into `cache` and
/// shared by every later scan of the operator.
fn scan_type(
    runtime: &Runtime,
    relationship_type: &Arc<String>,
    cache: &mut Option<Arc<Vec<EdgeHit>>>,
) -> impl Iterator<Item = EdgeHit> + use<> {
    let edges =
        cache.get_or_insert_with(|| Arc::new(runtime.graph().get_all_edges(relationship_type)));
    shared_hits(Arc::clone(edges))
}

/// The edges the index returns for `q` (which `can_utilize_index` accepted), in graph
/// order `(src, dst, edge)`.
#[cfg_attr(
    feature = "index-falkordb",
    allow(
        clippy::needless_pass_by_value,
        reason = "the native index only borrows the query; the default index consumes it"
    )
)]
#[cfg_attr(
    not(feature = "index-falkordb"),
    allow(
        clippy::unnecessary_wraps,
        reason = "only the native index can refuse a query, but both builds share the signature"
    )
)]
#[cfg_attr(
    not(feature = "index-falkordb"),
    allow(
        clippy::needless_pass_by_ref_mut,
        reason = "only the native index's still-building scan fills the cache, but both builds \
                  share the signature"
    )
)]
fn indexed_edges<'r>(
    runtime: &'r Runtime,
    relationship_type: &Arc<String>,
    all_edges_cache: &mut Option<Arc<Vec<EdgeHit>>>,
    q: IndexQuery<Value>,
) -> Result<Box<dyn Iterator<Item = EdgeHit> + 'r>, String> {
    let g = runtime.graph();
    // Edge index: a numeric Equal/Range on a column we own is served
    // by the CoW B-tree (endpoints recovered from the graph's reverse index);
    // anything else is an error under the flag. Mirrors `NodeByIndexScanOp`.
    #[cfg(feature = "index-falkordb")]
    let it: Box<dyn Iterator<Item = EdgeHit> + 'r> =
        match g.query_index_edges(relationship_type, &q) {
            Some(Ok(rows)) => Box::new(rows),
            // Still building: scan the type for what the column will answer,
            // checking the predicate the planner dropped. See `NodeByIndexScanOp`.
            Some(Err(scan_filter)) => {
                // Resolved once per scan. No edge holds a name the graph has never seen.
                match scan_filter.resolve(|attr| {
                    g.get_relationship_attribute_id(attr)
                        .map(|attr| attr as u16)
                }) {
                    Some(scan_filter) => Box::new(
                        scan_type(runtime, relationship_type, all_edges_cache).filter(
                            move |&(_, _, edge)| {
                                runtime.graph().relationship_matches(edge, &scan_filter)
                            },
                        ),
                    ),
                    None => Box::new(std::iter::empty()),
                }
            }
            None => {
                // See node_by_index_scan: native is the only index under this flag, so an
                // unserviceable predicate is a loud error.
                return Err(unsupported_by_native_index(
                    "relationship",
                    relationship_type,
                    &q,
                ));
            }
        };
    #[cfg(not(feature = "index-falkordb"))]
    let it: Box<dyn Iterator<Item = EdgeHit> + 'r> = {
        let _ = all_edges_cache;
        Box::new(g.get_indexed_edges(relationship_type, q))
    };
    Ok(it)
}

/// Most hits (both orientations of an undirected pattern counted) a hit cache may
/// keep. A kept hit costs about 48 bytes (the list plus its place in an endpoint
/// bucket), so this is some 50 MB; a query with more hits streams the index per
/// row, as it did before the cache, and its memory stays that of a scan.
const MAX_CACHED_HITS: usize = 1 << 20;

/// `hits`, oriented as the pattern matches them (an undirected pattern lists an edge
/// in both orientations, a self-loop once), or `None` when there are more than `max`.
fn collect_hits(
    hits: impl Iterator<Item = EdgeHit>,
    bidirectional: bool,
    max: usize,
) -> Option<Vec<EdgeHit>> {
    let mut all = Vec::new();
    for (src, dst, edge) in hits {
        // Checked before each append, so an oversized query is refused without
        // the vector growing past the budget.
        if all.len() == max {
            return None;
        }
        all.push((src, dst, edge));
        if bidirectional && src != dst {
            if all.len() == max {
                return None;
            }
            all.push((dst, src, edge));
        }
    }
    Some(all)
}

/// Where a [`HitCache`] stands.
enum HitCacheState {
    /// Never kept: the query writes (an upstream `CALL {}` commits between
    /// batches, so kept hits would go stale), or the index query reads the input
    /// row, so rows do not all ask the index the same question.
    Off,
    /// No row with a bound endpoint has run yet.
    Cold,
    /// One bound row ran and streamed, so a one-row input pays no setup. The
    /// next bound row collects the hits: `Built`, or `OverBudget` if they do not
    /// fit.
    Warm,
    /// The hits of the index query, which every row asks.
    Built(IndexedHits),
    /// The hits did not fit in [`MAX_CACHED_HITS`]: every row streams the index,
    /// as before the cache existed, and memory stays that of a scan.
    OverBudget,
}

/// The hits of a row-invariant index query, kept so each bound input row looks
/// its endpoint up instead of re-running the index query and filtering every
/// hit.
struct HitCache {
    state: HitCacheState,
    transposed: bool,
    bidirectional: bool,
}

impl HitCache {
    /// The kept hits whose pattern `from` (else `to`) is the row's bound one, in
    /// list order. `None` when the row must stream the index instead: no
    /// endpoint is bound, the cache is off or over budget, or this is the first
    /// bound row. `q` is the row's index query; with the cache on it is every
    /// row's.
    fn lookup(
        &mut self,
        runtime: &Runtime,
        relationship_type: &Arc<String>,
        all_edges_cache: &mut Option<Arc<Vec<EdgeHit>>>,
        q: &IndexQuery<Value>,
        bound_from: Option<NodeId>,
        bound_to: Option<NodeId>,
    ) -> Result<Option<Arc<[EdgeHit]>>, String> {
        let (node, by_from) = match (bound_from, bound_to) {
            (Some(id), _) => (id, true),
            (None, Some(id)) => (id, false),
            (None, None) => return Ok(None),
        };
        if matches!(self.state, HitCacheState::Cold) {
            self.state = HitCacheState::Warm;
            return Ok(None);
        }
        if matches!(self.state, HitCacheState::Warm) {
            let hits = indexed_edges(runtime, relationship_type, all_edges_cache, q.clone())?;
            self.state = collect_hits(hits, self.bidirectional, MAX_CACHED_HITS).map_or(
                HitCacheState::OverBudget,
                |all| {
                    HitCacheState::Built(IndexedHits {
                        all,
                        by_from: None,
                        by_to: None,
                        no_hits: Arc::from(Vec::new()),
                    })
                },
            );
        }
        let HitCacheState::Built(hits) = &mut self.state else {
            return Ok(None);
        };
        let groups = if by_from {
            &mut hits.by_from
        } else {
            &mut hits.by_to
        };
        let groups =
            groups.get_or_insert_with(|| group_by_endpoint(&hits.all, self.transposed, by_from));
        Ok(Some(
            groups
                .get(&node)
                .map_or_else(|| Arc::clone(&hits.no_hits), Arc::clone),
        ))
    }
}

/// Invariant: `relationship.from` is always the bound endpoint of the
/// edge scan.
///
/// `transposed` flips which graph-side endpoint the edge
/// tensor keys on (src vs dst) but the bound endpoint is always read
/// via `rp.from.alias`.
///
/// The planner's `select_scan_node` pass enforces
/// this by swapping the relationship and setting `transposed`
/// accordingly before any rewrite to `EdgeByIndexScan`.
pub struct EdgeByIndexScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    transposed: bool,
    /// Holds the parent batch being expanded and the per-row edge iterators, and
    /// performs the shared pack-and-gather emit. Each pushed iterator yields
    /// `(src, dst, edge)` in graph-tensor order; the emitter's binding maps the
    /// graph sides onto the pattern's from/to endpoints (honoring `transposed`)
    /// and skips the second endpoint for self-loop patterns.
    emitter: BatchedResultEmitter<'a, (NodeId, NodeId, RelationshipId)>,
    /// Lazily-populated cache of all edges of `relationship_pattern.types[0]`
    /// for the non-indexable-runtime-value fallback. Materialized once
    /// on the first fallback and shared across subsequent input rows
    /// so we don't pay O(|E_type|) per row when the index can't serve
    /// the query (e.g. value is a list or date). The cache is
    /// dropped when the op is dropped.
    all_edges_cache: Option<Arc<Vec<EdgeHit>>>,
    /// Kept hits of a row-invariant index query; see [`HitCache`].
    hit_cache: HitCache,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> EdgeByIndexScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        query: &'a IndexQuery<QueryExpr<Variable>>,
        transposed: bool,
        idx: NodeIdx<Dyn<IR>>,
        // Kept in the signature, unused for now: see the emitter below.
        _record_cap: Option<usize>,
    ) -> Self {
        // Self-loop patterns like `MATCH (n)-[r:T]->(n)` share one alias on both
        // endpoints; bind it once (via `from`) and skip the `to` column so the
        // second insert can't overwrite the first.
        let to = (relationship_pattern.to.alias != relationship_pattern.from.alias)
            .then_some(relationship_pattern.to.alias.id);
        Self {
            runtime,
            child,
            // `None`, for the same reason as `node_by_index_scan`: capping an
            // index scan hangs `tests/flow/test_constraint.py`, and this line
            // does it on its own. See the comment there.
            emitter: BatchedResultEmitter::with_binding(
                EdgeEndpoints {
                    from: relationship_pattern.from.alias.id,
                    to,
                    edge: relationship_pattern.alias.id,
                    transposed,
                },
                None,
            ),
            relationship_pattern,
            query,
            transposed,
            all_edges_cache: None,
            hit_cache: HitCache {
                state: if runtime.write || !query_is_row_invariant(query) {
                    HitCacheState::Off
                } else {
                    HitCacheState::Cold
                },
                transposed,
                bidirectional: relationship_pattern.bidirectional,
            },
            idx,
        }
    }

    fn evaluate_index_query<R: crate::runtime::row::RowView + ?Sized>(
        runtime: &Runtime,
        query: &IndexQuery<QueryExpr<Variable>>,
        vars: &R,
    ) -> Result<IndexQuery<Value>, String> {
        match query {
            IndexQuery::Equal { key, value } => {
                let value = {
                    ExprEval::from_runtime(runtime).eval(
                        value,
                        value.root().idx(),
                        Some(vars),
                        None,
                    )
                }?;
                Ok(IndexQuery::Equal {
                    key: key.clone(),
                    value,
                })
            }
            IndexQuery::Range {
                key,
                min,
                max,
                include_min,
                include_max,
            } => {
                let (min, max) = match (min, max) {
                    (Some(min), Some(max)) => {
                        let min = ExprEval::from_runtime(runtime).eval(
                            min,
                            min.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        let max = ExprEval::from_runtime(runtime).eval(
                            max,
                            max.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (Some(min), Some(max))
                    }
                    (Some(min), None) => {
                        let min = ExprEval::from_runtime(runtime).eval(
                            min,
                            min.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (Some(min), None)
                    }
                    (None, Some(max)) => {
                        let max = ExprEval::from_runtime(runtime).eval(
                            max,
                            max.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (None, Some(max))
                    }
                    (None, None) => (None, None),
                };
                Ok(IndexQuery::Range {
                    key: key.clone(),
                    min,
                    max,
                    include_min: *include_min,
                    include_max: *include_max,
                })
            }
            IndexQuery::Point { key, point, radius } => {
                let point = ExprEval::from_runtime(runtime).eval(
                    point,
                    point.root().idx(),
                    Some(vars),
                    None,
                )?;
                let radius = ExprEval::from_runtime(runtime).eval(
                    radius,
                    radius.root().idx(),
                    Some(vars),
                    None,
                )?;
                Ok(IndexQuery::Point {
                    key: key.clone(),
                    point,
                    radius,
                })
            }
            IndexQuery::And(queries) => {
                let mut evaluated = Vec::with_capacity(queries.len());
                for q in queries {
                    evaluated.push(Self::evaluate_index_query(runtime, q, vars)?);
                }
                Ok(IndexQuery::And(evaluated))
            }
            IndexQuery::Or(queries) => {
                let mut evaluated = Vec::with_capacity(queries.len());
                for q in queries {
                    evaluated.push(Self::evaluate_index_query(runtime, q, vars)?);
                }
                Ok(IndexQuery::Or(evaluated))
            }
            IndexQuery::ArrayContains { key, value } => {
                let value = {
                    ExprEval::from_runtime(runtime).eval(
                        value,
                        value.root().idx(),
                        Some(vars),
                        None,
                    )
                }?;
                Ok(IndexQuery::ArrayContains {
                    key: key.clone(),
                    value,
                })
            }
            IndexQuery::InList { key, list } => {
                let list_val = {
                    ExprEval::from_runtime(runtime).eval(list, list.root().idx(), Some(vars), None)
                }?;
                match list_val {
                    // Kept as one `InList`, as in `NodeByIndexScanOp`: one attribute by
                    // construction, and RediSearch builds the same union from either shape.
                    Value::List(items) => {
                        let keep = |v: &Value| {
                            matches!(
                                v,
                                Value::Int(_) | Value::Float(_) | Value::String(_) | Value::Bool(_)
                            )
                        };
                        let items = if items.iter().all(keep) {
                            items
                        } else {
                            Arc::new(items.iter().filter(|v| keep(v)).cloned().collect())
                        };
                        Ok(IndexQuery::InList {
                            key: key.clone(),
                            list: Value::List(items),
                        })
                    }
                    _ => Err("IN operator requires a list".into()),
                }
            }
        }
    }

    /// Check if an evaluated index query can be satisfied by the index.
    fn can_utilize_index(q: &IndexQuery<Value>) -> bool {
        use crate::index::Index;

        const fn is_indexable(v: &Value) -> bool {
            match v {
                Value::Int(i) => !Index::int_loses_f64_precision(*i),
                Value::Float(_)
                | Value::String(_)
                | Value::Bool(_)
                | Value::Point(_)
                | Value::Null => true,
                _ => false,
            }
        }
        match q {
            IndexQuery::Equal { value, .. } => is_indexable(value),
            IndexQuery::Range { min, max, .. } => {
                min.as_ref().is_none_or(is_indexable) && max.as_ref().is_none_or(is_indexable)
            }
            IndexQuery::And(children) | IndexQuery::Or(children) => {
                // Empty `Or([])` / `And([])` is not a valid index
                // query: the RediSearch backend treats it as
                // match-all, which is unsafe when the optimizer has
                // already pushed an IN-list whose elements were all
                // filtered out at runtime (e.g. `IN [NULL]`). Force
                // the fallback path so the retained post-filter
                // re-establishes correctness.
                !children.is_empty() && children.iter().all(Self::can_utilize_index)
            }
            // The same rule, for an `IN` list: empty once its members were filtered (`IN [NULL]`)
            // falls back, and so does a list with any member the index cannot hold exactly.
            IndexQuery::InList { list, .. } => matches!(
                list,
                Value::List(items) if !items.is_empty() && items.iter().all(is_indexable)
            ),
            IndexQuery::ArrayContains { value, .. } => match value {
                // Match `Equal` / `Range`: reject int64s that lose
                // f64 precision — the index can't represent them
                // exactly so the query must fall back to post-filter.
                Value::Int(i) => !Index::int_loses_f64_precision(*i),
                Value::Float(_) | Value::String(_) | Value::Bool(_) => true,
                _ => false,
            },
            IndexQuery::Point { .. } => true,
        }
    }
}

impl<'a> Iterator for EdgeByIndexScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // One (endpoint-filtered) edge iterator per active parent row, built
            // lazily one row at a time. When the batch is exhausted (`Ok(None)`),
            // pull and seed the next child batch.
            let relationship_type = &self.relationship_pattern.types[0];
            let rp = self.relationship_pattern;
            let runtime = self.runtime;
            // The closure borrows these two fields mutably while `emit_lazy`
            // borrows the emitter: disjoint fields, so no cell is needed.
            let hit_cache = &mut self.hit_cache;
            let all_edges_cache = &mut self.all_edges_cache;
            match self.emitter.emit_lazy(|b, row| {
                let view = BatchRow::new(b, row);
                let q = Self::evaluate_index_query(runtime, self.query, &view)?;

                // Stream results instead of collecting: the child
                // batch may have many rows and each row would
                // otherwise materialize the full edge set before any
                // emission. Matches `NodeByIndexScanOp`'s pattern.
                //
                // On the fallback path (non-indexable runtime value),
                // `get_all_edges` is materialized once into the op's
                // cache and shared by all subsequent rows via `Arc`,
                // so we don't rebuild the full-type Vec per row.
                let bound_from = match view.value_at(rp.from.alias.id) {
                    Some(Value::Node(id)) => Some(id),
                    _ => None,
                };
                let bound_to = match view.value_at(rp.to.alias.id) {
                    Some(Value::Node(id)) => Some(id),
                    _ => None,
                };
                let filter = EndpointFilter {
                    bound_from,
                    bound_to,
                    same_endpoint_alias: rp.from.alias == rp.to.alias,
                    transposed: self.transposed,
                };
                let usable = Self::can_utilize_index(&q);
                if usable
                    && let Some(hits) = hit_cache.lookup(
                        runtime,
                        relationship_type,
                        all_edges_cache,
                        &q,
                        bound_from,
                        bound_to,
                    )?
                {
                    // An endpoint with no kept hits yields nothing: skip the row
                    // instead of boxing an empty iterator.
                    if hits.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some(RowIter::many(Box::new(
                        shared_hits(hits).filter(move |&hit| filter.keeps(hit)),
                    ))));
                }
                // An index-unshaped predicate is scanned instead; the planner keeps a
                // filter for it. Mirrors `NodeByIndexScanOp`.
                let base: Box<dyn Iterator<Item = EdgeHit>> = if usable {
                    indexed_edges(runtime, relationship_type, all_edges_cache, q)?
                } else {
                    Box::new(scan_type(runtime, relationship_type, all_edges_cache))
                };

                // An undirected pattern matches each edge in both
                // orientations (a self-loop once), as `CondTraverse` does;
                // the index yields it once, `src -> dst`.
                let oriented_hits: Box<dyn Iterator<Item = EdgeHit>> = if rp.bidirectional {
                    Box::new(base.flat_map(|(src, dst, edge)| {
                        std::iter::once((src, dst, edge))
                            .chain((src != dst).then_some((dst, src, edge)))
                    }))
                } else {
                    base
                };

                // Filter edges by *both* endpoints when the child has
                // already bound them; see `EndpointFilter`.
                let edges: Box<dyn Iterator<Item = EdgeHit>> = if filter.keeps_all() {
                    oriented_hits
                } else {
                    Box::new(oriented_hits.filter(move |&hit| filter.keeps(hit)))
                };
                Ok(Some(RowIter::many(edges)))
            }) {
                Ok(Some(out)) => return Some(Ok(out)),
                Ok(None) => match self.child.next() {
                    Some(Ok(batch)) => self.emitter.seed(batch),
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn in_list(values: &[Value]) -> IndexQuery<Value> {
        IndexQuery::InList {
            key: key(),
            list: Value::List(Arc::new(values.iter().cloned().collect())),
        }
    }

    /// The same members as an `Or` of `Equal`s — the shape an `IN` list used to reach the index in.
    fn or_of_equals(values: &[Value]) -> IndexQuery<Value> {
        IndexQuery::Or(
            values
                .iter()
                .map(|v| IndexQuery::Equal {
                    key: key(),
                    value: v.clone(),
                })
                .collect(),
        )
    }

    /// The hit cache keeps at most `max` hits and says so by returning `None`, so a
    /// query with more hits streams instead of holding them all.
    #[test]
    fn collect_hits_respects_its_budget_and_orients_undirected_hits() {
        let hits = |n: u64| {
            (0..n).map(|i| {
                (
                    NodeId::from(i),
                    NodeId::from(i + 1),
                    RelationshipId::from(i),
                )
            })
        };
        assert_eq!(collect_hits(hits(5), false, 5).map(|v| v.len()), Some(5));
        assert!(collect_hits(hits(6), false, 5).is_none());
        // an undirected pattern counts both orientations against the budget
        assert_eq!(collect_hits(hits(3), true, 6).map(|v| v.len()), Some(6));
        assert!(collect_hits(hits(4), true, 6).is_none());
        // a self-loop is one hit
        let one_loop = std::iter::once((
            NodeId::from(7u64),
            NodeId::from(7u64),
            RelationshipId::from(0u64),
        ));
        assert_eq!(collect_hits(one_loop, true, 1).map(|v| v.len()), Some(1));
        assert_eq!(
            collect_hits(std::iter::empty(), false, 0).map(|v| v.len()),
            Some(0)
        );
    }

    /// An `IN` list reaches the index only when it is non-empty and every member is one the index
    /// holds exactly; otherwise the op scans. That is the rule its members got as an `Or` of
    /// `Equal`s, so which queries fall back is unchanged by handing the list over whole.
    #[test]
    fn in_list_reaches_the_index_on_the_same_terms_as_an_or_of_its_members() {
        let lossy = Value::Int(9_007_199_254_740_993);
        let cases: [(&[Value], bool); 5] = [
            (&[Value::Int(1), Value::Float(2.5), Value::Bool(true)], true),
            (
                &[Value::Int(1), Value::String(Arc::new("a".to_string()))],
                true,
            ),
            (&[], false),
            (&[Value::Int(1), lossy.clone()], false),
            (&[lossy], false),
        ];
        for (values, expected) in cases {
            assert_eq!(
                EdgeByIndexScanOp::can_utilize_index(&in_list(values)),
                expected,
                "IN {values:?}"
            );
            assert_eq!(
                EdgeByIndexScanOp::can_utilize_index(&in_list(values)),
                EdgeByIndexScanOp::can_utilize_index(&or_of_equals(values)),
                "IN {values:?} must decide as its members did as an Or"
            );
        }
        assert!(
            !EdgeByIndexScanOp::can_utilize_index(&IndexQuery::InList {
                key: key(),
                list: Value::Int(1),
            }),
            "an InList that does not hold a list never reaches the index"
        );
    }
}
