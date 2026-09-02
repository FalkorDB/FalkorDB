//! Columnar evaluation of whole expression trees.
//!
//! [`vectorized`](super::vectorized) compares one materialized column against
//! one constant. This module is the layer above: it walks an `ExprIR` tree and
//! evaluates *every* node over a batch of rows at once, so an arbitrary
//! predicate — `p.age % 7 = 3`, `p.age > 70 OR p.score < 100.0`,
//! `toUpper(p.name) = 'P100'` — pays one bulk property fetch and one pass per
//! operator instead of one tree walk per row.
//!
//! ```text
//!   per-row                            columnar
//!   =======                            ========
//!   for each row:                      age  = bulk fetch p.age      (1 call)
//!     fetch p.age                      t0   = age % 7               (1 pass)
//!     eval  % 7                        mask = t0 == 3               (1 pass)
//!     eval  == 3
//!   O(rows x tree)                     O(rows) per operator
//! ```
//!
//! ## Total, not partial
//!
//! Every node evaluates to a [`ExprColumn`]; a node this module has no columnar
//! arm for is evaluated per row into [`ExprColumn::Values`] and the walk continues
//! columnar above it. So an unsupported operator costs what it costs today
//! while its operands — property reads especially — still go through the bulk
//! path. There is no "this tree is not vectorizable" answer to fall back on.
//!
//! ## Agreement with the per-row evaluator
//!
//! Same rule as the comparison kernels: a typed lane is entered only when its
//! primitive operation *is* the [`Value`] operation (`Int`/`Int` addition is
//! `wrapping_add` on both paths, and so on), and every other shape is computed
//! by calling the very same `Value` operator per element. Nulls and the
//! three-valued logic of `AND`/`OR`/`XOR`/`NOT` are tracked explicitly, which
//! `ExprColumn::Bools` carries as a null bitmap beside its mask.
//!
//! Short-circuiting is preserved by *narrowing rows*: `A AND B` evaluates `B`
//! only over the rows where `A` was not `false`, exactly as the per-row
//! evaluator stops at the first `false` child. Without that, `WHERE p.age > 5
//! AND 1 / p.zero = 1` would raise a division error on rows the per-row path
//! never evaluates. `CASE` narrows the same way, one branch at a time.

use std::sync::Arc;

use crate::parser::ast::{ExprIR, Variable};
use crate::runtime::batch::{Batch, BatchRow, Column, NullBitmap, classify_exact_column};
use crate::runtime::eval::{ExprEval, ExprNode};
use crate::runtime::functions::FnType;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use crate::runtime::vectorized::{
    CmpOp, compare_f64_column, compare_i64_column, compare_value_column_3vl, compare_values,
};

use orx_tree::NodeRef;

// ---------------------------------------------------------------------------
// ExprColumn — one expression's value across a batch of rows
// ---------------------------------------------------------------------------

/// The value of one expression over `rows`, entry `i` answering `rows[i]`.
///
/// The typed variants carry a null bitmap because a primitive lane has no
/// in-band null; [`ExprColumn::Values`] carries `Value::Null` in-band instead.
#[derive(Debug)]
pub enum ExprColumn {
    /// One value shared by every row — constants and parameters, which never
    /// need materializing.
    Scalar(Value),
    Ints(Vec<i64>, NullBitmap),
    Floats(Vec<f64>, NullBitmap),
    /// Three-valued booleans: `data[i]` is meaningful only where not null.
    Bools(Vec<bool>, NullBitmap),
    Values(Vec<Value>),
}

impl ExprColumn {
    /// The value at position `i`.
    #[must_use]
    pub fn get(
        &self,
        i: usize,
    ) -> Value {
        match self {
            Self::Scalar(v) => v.clone(),
            Self::Ints(data, nulls) => {
                if nulls.is_null(i) {
                    Value::Null
                } else {
                    Value::Int(data[i])
                }
            }
            Self::Floats(data, nulls) => {
                if nulls.is_null(i) {
                    Value::Null
                } else {
                    Value::Float(data[i])
                }
            }
            Self::Bools(data, nulls) => {
                if nulls.is_null(i) {
                    Value::Null
                } else {
                    Value::Bool(data[i])
                }
            }
            Self::Values(data) => data[i].clone(),
        }
    }

    /// Materializes `len` rows as `Value`s — what a projection emits and what
    /// the generic operator lanes consume.
    #[must_use]
    pub fn into_values(
        self,
        len: usize,
    ) -> Vec<Value> {
        match self {
            Self::Values(data) => data,
            Self::Scalar(v) => vec![v; len],
            other => (0..len).map(|i| other.get(i)).collect(),
        }
    }

    /// This column's null bitmap, or an empty one for the variants that carry
    /// their nulls in band.
    fn nulls_or_none(
        &self,
        len: usize,
    ) -> NullBitmap {
        match self {
            Self::Ints(_, nulls) | Self::Floats(_, nulls) | Self::Bools(_, nulls) => nulls.clone(),
            _ => NullBitmap::none(len),
        }
    }

    /// `Some(nullness)` when every row shares it — a constant, or a typed lane
    /// with an empty bitmap. `None` when it varies per row.
    fn constant_nullness(&self) -> Option<bool> {
        match self {
            Self::Scalar(v) => Some(matches!(v, Value::Null)),
            Self::Ints(_, nulls) | Self::Floats(_, nulls) | Self::Bools(_, nulls) => {
                (!nulls.any_null()).then_some(false)
            }
            Self::Values(_) => None,
        }
    }

    /// True when row `i` is null.
    fn is_null(
        &self,
        i: usize,
    ) -> bool {
        match self {
            Self::Scalar(v) => matches!(v, Value::Null),
            Self::Ints(_, nulls) | Self::Floats(_, nulls) | Self::Bools(_, nulls) => {
                nulls.is_null(i)
            }
            Self::Values(data) => matches!(data[i], Value::Null),
        }
    }
}

/// Union of two operands' nulls — the shape every binary typed lane needs.
///
/// A constant operand is null for every row or for none, which is the common
/// case (`p.age > 45`) and settles the union without a row loop.
fn union_nulls(
    lhs: &ExprColumn,
    rhs: &ExprColumn,
    len: usize,
) -> NullBitmap {
    match (lhs.constant_nullness(), rhs.constant_nullness()) {
        // A null constant makes every row null, whatever the other side holds.
        (Some(true), _) | (_, Some(true)) => NullBitmap::all(len),
        (Some(false), Some(false)) => NullBitmap::none(len),
        (Some(false), None) => rhs.nulls_or_none(len),
        (None, Some(false)) => lhs.nulls_or_none(len),
        _ => {
            let mut bitmap = NullBitmap::none(len);
            for i in 0..len {
                if lhs.is_null(i) || rhs.is_null(i) {
                    bitmap.set(i);
                }
            }
            bitmap
        }
    }
}

// ---------------------------------------------------------------------------
// VectorEval — the tree walk
// ---------------------------------------------------------------------------

/// Evaluates expression trees over a batch of rows.
pub struct VectorEval<'a> {
    runtime: &'a Runtime<'a>,
}

impl<'a> VectorEval<'a> {
    #[must_use]
    pub const fn new(runtime: &'a Runtime<'a>) -> Self {
        Self { runtime }
    }

    /// Evaluates `node` over `rows` of `batch`.
    ///
    /// The result is aligned to `rows`: entry `i` is the value for row
    /// `rows[i]`. Never fails to produce a column — a shape with no columnar
    /// arm is evaluated per row into [`ExprColumn::Values`].
    pub fn eval(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Result<ExprColumn, String> {
        let len = rows.len();
        match node.data() {
            ExprIR::Constant(v) => Ok(ExprColumn::Scalar(v.clone())),
            ExprIR::Parameter(name) => self
                .runtime
                .parameters
                .get(name)
                .map(|v| ExprColumn::Scalar(v.clone()))
                .ok_or_else(|| format!("Parameter {name} not found")),
            ExprIR::Variable(var) => Ok(self.eval_variable(var, batch, rows)),
            ExprIR::Property(attr) if node.num_children() == 1 => {
                match self.eval_property(attr, &node.child(0), batch, rows) {
                    Some(vector) => Ok(vector),
                    None => self.eval_per_row(node, batch, rows),
                }
            }
            ExprIR::Paren if node.num_children() == 1 => self.eval(&node.child(0), batch, rows),
            ExprIR::And => self.eval_and_or(node, batch, rows, false),
            ExprIR::Or => self.eval_and_or(node, batch, rows, true),
            ExprIR::Not if node.num_children() == 1 => {
                let child = self.eval(&node.child(0), batch, rows)?;
                negate_bools(&child, len)
            }
            op @ (ExprIR::Add | ExprIR::Sub | ExprIR::Mul | ExprIR::Div | ExprIR::Modulo)
                if node.num_children() == 2 =>
            {
                let lhs = self.eval(&node.child(0), batch, rows)?;
                let rhs = self.eval(&node.child(1), batch, rows)?;
                arithmetic(op, lhs, rhs, len)
            }
            ExprIR::Case { has_subject } => self.eval_case(*has_subject, node, batch, rows),
            op if node.num_children() == 2 && CmpOp::from_expr_ir(op).is_some() => {
                let cmp = CmpOp::from_expr_ir(op).expect("guarded by the match arm");
                let lhs = self.eval(&node.child(0), batch, rows)?;
                let rhs = self.eval(&node.child(1), batch, rows)?;
                Ok(compare_columns(&lhs, &rhs, cmp, len))
            }
            ExprIR::FuncInvocation(func)
                if !matches!(func.fn_type, FnType::Aggregation { .. })
                    && func.struct_fn.is_none()
                    && !func.write
                    && !node
                        .children()
                        .any(|c| matches!(c.data(), ExprIR::Distinct)) =>
            {
                if let Some(column) = (func.name == "hasLabels")
                    .then(|| self.eval_has_labels(node, batch, rows))
                    .flatten()
                {
                    return Ok(column);
                }
                self.eval_function(node, batch, rows)
            }
            _ => self.eval_per_row(node, batch, rows),
        }
    }

    /// Evaluates `node` and hands back plain `Value`s — what a projection
    /// emits and what an aggregate accumulates.
    ///
    /// This is not `eval(..).into_values(..)`: the two leading arms exist to
    /// *skip* [`eval`](Self::eval) for the shapes whose result is already a
    /// `Value`. A property read and a variable passthrough would otherwise be
    /// classified into a typed column and rebuilt back into `Value`s — two
    /// extra passes and an allocation for nothing, because no operator above
    /// is going to consume the typed lane. Routing `ProjectOp` through
    /// `eval(..).into_values(..)` instead measured 5,674,324 instructions on
    /// the `RETURN DISTINCT` benchmark row against 4,988,111 with these arms:
    /// a 1.14x regression on projections that read a property.
    pub fn eval_values(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Result<Vec<Value>, String> {
        match node.data() {
            ExprIR::Property(attr) if node.num_children() == 1 => {
                if let Some(values) = self.property_values(attr, &node.child(0), batch, rows) {
                    return Ok(values);
                }
            }
            ExprIR::Variable(var) => {
                return Ok(rows
                    .iter()
                    .map(|&row| batch.value_at(var.id, row).unwrap_or(Value::Null))
                    .collect());
            }
            _ => {}
        }
        Ok(self.eval(node, batch, rows)?.into_values(rows.len()))
    }

    /// A bound variable: gathered from the batch column it lives in.
    fn eval_variable(
        &self,
        var: &Variable,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> ExprColumn {
        match batch.column(var.id) {
            Column::Ints(data) => {
                let gathered: Vec<i64> = rows.iter().map(|&r| data[r]).collect();
                let nulls = NullBitmap::none(gathered.len());
                ExprColumn::Ints(gathered, nulls)
            }
            Column::Floats(data) => {
                let gathered: Vec<f64> = rows.iter().map(|&r| data[r]).collect();
                let nulls = NullBitmap::none(gathered.len());
                ExprColumn::Floats(gathered, nulls)
            }
            col => ExprColumn::Values(rows.iter().map(|&r| col.get(r)).collect()),
        }
    }

    /// `var.attr` over a node or relationship column: one bulk attribute fetch.
    /// `None` when the child is not a bound entity column, so the caller falls
    /// back to per-row evaluation.
    fn eval_property(
        &self,
        attr: &Arc<String>,
        child: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Option<ExprColumn> {
        let values = self.property_values(attr, child, batch, rows)?;
        let (col, nulls) = classify_exact_column(values);
        Some(match col {
            Column::Ints(data) => ExprColumn::Ints(data, nulls),
            Column::Floats(data) => ExprColumn::Floats(data, nulls),
            // `classify_exact_column` hands the original vector back untouched
            // when no typed lane fits, so take it rather than cloning every
            // string, list and map out of it again.
            Column::Values(data) => ExprColumn::Values(data),
            other => ExprColumn::Values((0..rows.len()).map(|i| other.get(i)).collect()),
        })
    }

    /// The stored values of `var.attr` for `rows`, in one bulk fetch. `None`
    /// when the child is not a bound node or relationship column, so the caller
    /// falls back to per-row evaluation.
    fn property_values(
        &self,
        attr: &Arc<String>,
        child: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Option<Vec<Value>> {
        let ExprIR::Variable(var) = child.data() else {
            return None;
        };
        match batch.column(var.id) {
            Column::NodeIds(ids) => {
                let gathered: Vec<_> = rows.iter().map(|&r| ids[r]).collect();
                Some(
                    self.runtime
                        .materialize_node_property_values(&gathered, attr),
                )
            }
            Column::RelIds(ids) => {
                let gathered: Vec<_> = rows.iter().map(|&r| ids[r]).collect();
                Some(
                    self.runtime
                        .materialize_relationship_property_values(&gathered, attr),
                )
            }
            _ => None,
        }
    }

    /// `AND` / `OR` with three-valued logic and per-row short-circuiting.
    ///
    /// `short_circuit_on` is the child result that settles a row: `false` ends
    /// an `AND`, `true` ends an `OR`. Settled rows are dropped from the row set
    /// handed to the remaining children, so a child is evaluated for exactly
    /// the rows the per-row evaluator would have reached.
    fn eval_and_or(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
        short_circuit_on: bool,
    ) -> Result<ExprColumn, String> {
        let len = rows.len();
        // Neutral element: `AND` starts true, `OR` starts false.
        let mut result = vec![!short_circuit_on; len];
        let mut nulls = NullBitmap::none(len);
        // Rows still undecided, as positions into `rows`.
        let mut live: Vec<usize> = (0..len).collect();

        let mut live_rows: Vec<usize> = Vec::new();
        // Reused across children so narrowing costs no allocation per conjunct.
        let mut still_live: Vec<usize> = Vec::with_capacity(len);
        for child in node.children() {
            if live.is_empty() {
                break;
            }
            // Until a row has been settled, `live` still covers every row, so
            // the child can read `rows` directly instead of a copy of it.
            let child_rows = if live.len() == len {
                rows
            } else {
                live_rows.clear();
                live_rows.extend(live.iter().map(|&i| rows[i]));
                &live_rows
            };
            let vector = self.eval(&child, batch, child_rows)?;
            still_live.clear();

            // A comparison child is already a mask: read it directly rather
            // than rebuilding a `Value` per row, which is the whole point of
            // the columnar path for `a > 1 AND b < 2`.
            if let ExprColumn::Bools(mask, child_nulls) = &vector {
                for (offset, &pos) in live.iter().enumerate() {
                    if child_nulls.is_null(offset) {
                        nulls.set(pos);
                        still_live.push(pos);
                    } else if mask[offset] == short_circuit_on {
                        result[pos] = short_circuit_on;
                        // A settled row wins outright, even over an earlier null.
                        nulls.clear(pos);
                    } else {
                        still_live.push(pos);
                    }
                }
                std::mem::swap(&mut live, &mut still_live);
                continue;
            }

            for (offset, &pos) in live.iter().enumerate() {
                match vector.get(offset) {
                    Value::Bool(b) if b == short_circuit_on => {
                        result[pos] = short_circuit_on;
                        nulls.clear(pos);
                    }
                    Value::Bool(_) => still_live.push(pos),
                    Value::Null => {
                        nulls.set(pos);
                        still_live.push(pos);
                    }
                    v => {
                        return Err(format!("Type mismatch: expected Bool but was {v:?}"));
                    }
                }
            }
            std::mem::swap(&mut live, &mut still_live);
        }

        Ok(ExprColumn::Bools(result, nulls))
    }

    /// `CASE`, one arm at a time over the rows that arm actually claims.
    ///
    /// Laziness is the point of the per-row `eval_case`: a `THEN` must not be
    /// evaluated for a row whose `WHEN` did not match, or
    /// `CASE WHEN n.d <> 0 THEN 1 / n.d ELSE 0 END` would divide by zero. The
    /// columnar form keeps that by narrowing: each branch expression is
    /// evaluated only over the rows it won, and the results are scattered back
    /// into their original positions.
    fn eval_case(
        &self,
        has_subject: bool,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Result<ExprColumn, String> {
        let len = rows.len();
        let subject = if has_subject {
            Some(self.eval(&node.child(0), batch, rows)?)
        } else {
            None
        };
        let arms = node.child(usize::from(has_subject));
        let num_arms = arms.num_children();

        let mut out = vec![Value::Null; len];
        // Positions into `rows` that no arm has claimed yet.
        let mut unclaimed: Vec<usize> = (0..len).collect();

        let mut arm = 0;
        while arm + 1 < num_arms && !unclaimed.is_empty() {
            let live_rows: Vec<usize> = unclaimed.iter().map(|&i| rows[i]).collect();
            let when = self.eval(&arms.child(arm), batch, &live_rows)?;

            let mut claimed = Vec::new();
            let mut still_unclaimed = Vec::with_capacity(unclaimed.len());
            for (offset, &pos) in unclaimed.iter().enumerate() {
                let matched = match &subject {
                    // Value form: the arm matches when it equals the subject.
                    // `Value`'s equality, deliberately — this engine matches a
                    // null subject against a `WHEN null` arm (pinned by
                    // test_function_calls.py's `test68_Case`), where openCypher
                    // would fall through to `ELSE`. The per-row `eval_case`
                    // does the same, and the two must not diverge.
                    Some(subject) => when.get(offset) == subject.get(pos),
                    // Searched form: anything but `false` / `null` matches, so
                    // a non-boolean condition is truthy rather than an error.
                    None => !matches!(when.get(offset), Value::Bool(false) | Value::Null),
                };
                if matched {
                    claimed.push(pos);
                } else {
                    still_unclaimed.push(pos);
                }
            }

            if !claimed.is_empty() {
                let claimed_rows: Vec<usize> = claimed.iter().map(|&i| rows[i]).collect();
                let then = self.eval(&arms.child(arm + 1), batch, &claimed_rows)?;
                for (offset, &pos) in claimed.iter().enumerate() {
                    out[pos] = then.get(offset);
                }
            }

            unclaimed = still_unclaimed;
            arm += 2;
        }

        // ELSE is always the last child; the parser substitutes a null constant
        // when the query omits one.
        if !unclaimed.is_empty() {
            let else_rows: Vec<usize> = unclaimed.iter().map(|&i| rows[i]).collect();
            let otherwise = self.eval(&node.child(node.num_children() - 1), batch, &else_rows)?;
            for (offset, &pos) in unclaimed.iter().enumerate() {
                out[pos] = otherwise.get(offset);
            }
        }

        Ok(ExprColumn::Values(out))
    }

    /// A scalar function call: operands are evaluated columnarly, then the
    /// function itself runs per row over the already-materialized arguments.
    /// The saving is the operand tree walk and the property fetches, not the
    /// call.
    fn eval_function(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Result<ExprColumn, String> {
        let ExprIR::FuncInvocation(func) = node.data() else {
            unreachable!("guarded by the caller")
        };
        let len = rows.len();
        let mut columns = Vec::with_capacity(node.num_children());
        for child in node.children() {
            columns.push(self.eval(&child, batch, rows)?);
        }

        // Every argument one value shared by the whole batch means the answer
        // is one value too — call the function once and hand back a `Scalar`
        // rather than the same call `len` times into a materialized column.
        //
        // This composes, which is the point. In
        // `split(replace(trim('  a,b,c  '), ',', ';'), ';')` the literal is a
        // `Scalar`, so `trim` returns one, so `replace` sees a `Scalar` and
        // returns one, and `split` likewise: the whole chain collapses to three
        // calls per batch from three per row. Measured on the
        // `split+trim+replace` benchmark row, 100 rows of exactly that:
        // 3,578 instructions per row in `split` alone before this.
        //
        // Three guards, and only the first is load-bearing today.
        //
        // Requiring an argument is what actually keeps `rand()` and
        // `randomUUID()` per row: they take none, so there would be nothing
        // here to prove them constant from, and folding would hand every row
        // in the batch one draw.
        //
        // `non_deterministic` is belt and braces on top of that. Every volatile
        // function in the registry is zero-argument or is flagged only for its
        // zero-argument "now" form — `date('2020-01-01')` answers the same
        // thing however often it is asked — so removing this check changes no
        // result today. It is here so that a volatile function which does take
        // an argument (a seeded `rand`, a `now(tz)`) cannot be added later and
        // silently start folding.
        //
        // The empty-batch check is the same kind of guard: operators drop empty
        // batches before reaching this, so a call that would raise is not
        // reached at zero rows either way. Stating it here keeps that a
        // property of this function rather than of every caller.
        if len > 0
            && !func.non_deterministic
            && !columns.is_empty()
            && columns.iter().all(|c| matches!(c, ExprColumn::Scalar(_)))
        {
            let args: Vec<Value> = columns.iter().map(|c| c.get(0)).collect();
            func.validate_args_type(&args)?;
            return Ok(ExprColumn::Scalar(func.func.call(self.runtime, &args)?));
        }

        let mut out = Vec::with_capacity(len);
        let mut args = Vec::with_capacity(columns.len());
        for i in 0..len {
            args.clear();
            args.extend(columns.iter().map(|c| c.get(i)));
            func.validate_args_type(&args)?;
            out.push(func.func.call(self.runtime, &args)?);
        }
        Ok(ExprColumn::Values(out))
    }

    /// `n:A:B` over a bound node column, as one boolean column.
    ///
    /// The generic function lane materializes a `Vec<Value>` of the operand and
    /// another of the result — 32 bytes a row before the call — and re-resolves
    /// every label *name* to its id on every row inside `hasLabels` itself. A
    /// label test needs neither: the names resolve once for the batch, and each
    /// row is then one bit of the label matrix written straight into a
    /// `Bools` column. `MATCH (n) WHERE n:Person RETURN count(n)` allocated
    /// 976 KB over 10k rows on the benchmark graph against 6 KB for the same
    /// scan filtering on a property.
    ///
    /// `None` whenever the shape is not exactly that — an unbound or
    /// heterogeneous operand column, or a label list that is not all string
    /// constants — and the caller takes the generic lane, which stays the
    /// definition of the answer.
    ///
    /// A name the graph has never registered is on no node, so an unresolvable
    /// label makes the whole conjunction false for every row. That is the same
    /// answer `Runtime::node_has_label` gives, reached without touching a
    /// matrix.
    fn eval_has_labels(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Option<ExprColumn> {
        if node.num_children() != 2 {
            return None;
        }
        let operand = node.child(0);
        let ExprIR::Variable(var) = operand.data() else {
            return None;
        };
        let Column::NodeIds(ids) = batch.column(var.id) else {
            // `Unbound` is all-null and `Values` may hold nulls or non-nodes;
            // both are the generic lane's business.
            return None;
        };

        let list = node.child(1);
        if !matches!(list.data(), ExprIR::List) || list.num_children() == 0 {
            return None;
        }
        // Two passes, and the order matters. `hasLabels` type-checks every
        // element of the list even once an earlier one has settled the answer,
        // so `hasLabels(n, ['NeverUsed', 1])` is a type error and not `false`.
        // Claiming the shape before checking the whole list would swallow that
        // error, which `test11_label_predicate_against_pending_labels` pins.
        let mut names = Vec::with_capacity(list.num_children());
        for child in list.children() {
            let ExprIR::Constant(Value::String(name)) = child.data() else {
                return None;
            };
            names.push(name.clone());
        }
        let mut label_ids = Vec::with_capacity(names.len());
        for name in &names {
            match self.runtime.label_id(name) {
                Some(id) => label_ids.push(id),
                // Unregistered: on no node, so false everywhere with no
                // per-row work at all.
                None => {
                    return Some(ExprColumn::Bools(
                        vec![false; rows.len()],
                        NullBitmap::none(rows.len()),
                    ));
                }
            }
        }

        let mut out = Vec::with_capacity(rows.len());
        for &row in rows {
            let id = ids[row];
            out.push(
                label_ids
                    .iter()
                    .all(|&label| self.runtime.node_has_label_id(id, label)),
            );
        }
        Some(ExprColumn::Bools(out, NullBitmap::none(rows.len())))
    }

    /// The universal fallback: evaluate this subtree once per row, through the
    /// same `BatchRow` view the per-row filter path uses.
    fn eval_per_row(
        &self,
        node: &ExprNode<'_>,
        batch: &Batch<'_>,
        rows: &[usize],
    ) -> Result<ExprColumn, String> {
        let eval = ExprEval::from_runtime(self.runtime);
        let mut out = Vec::with_capacity(rows.len());
        for &row in rows {
            let view = BatchRow::new(batch, row);
            out.push(eval.eval_node(node, Some(&view), None)?);
        }
        Ok(ExprColumn::Values(out))
    }
}

// ---------------------------------------------------------------------------
// Operator lanes
// ---------------------------------------------------------------------------

/// Compares two vectors with three-valued results.
///
/// Typed lanes are entered only where the primitive comparison is the `Value`
/// comparison; everything else goes through the same `compare_value` /
/// `partial_cmp` the per-row evaluator uses.
fn compare_columns(
    lhs: &ExprColumn,
    rhs: &ExprColumn,
    op: CmpOp,
    len: usize,
) -> ExprColumn {
    let nulls = union_nulls(lhs, rhs, len);
    let mask = match (lhs, rhs) {
        (ExprColumn::Ints(data, _), ExprColumn::Scalar(Value::Int(t))) => {
            Some(compare_i64_column(data, op, *t, &nulls))
        }
        (ExprColumn::Scalar(Value::Int(t)), ExprColumn::Ints(data, _)) => {
            Some(compare_i64_column(data, op.flip(), *t, &nulls))
        }
        (ExprColumn::Floats(data, _), ExprColumn::Scalar(Value::Float(t))) => {
            Some(compare_f64_column(data, op, *t, &nulls))
        }
        (ExprColumn::Scalar(Value::Float(t)), ExprColumn::Floats(data, _)) => {
            Some(compare_f64_column(data, op.flip(), *t, &nulls))
        }
        // `Value::compare_value` compares `Int` against `Float` as `i as f64`,
        // so the promoted lane is the same comparison.
        (ExprColumn::Ints(data, _), ExprColumn::Scalar(Value::Float(t))) => {
            let floats: Vec<f64> = data.iter().map(|&i| i as f64).collect();
            Some(compare_f64_column(&floats, op, *t, &nulls))
        }
        (ExprColumn::Floats(data, _), ExprColumn::Scalar(Value::Int(t))) => {
            Some(compare_f64_column(data, op, *t as f64, &nulls))
        }
        (ExprColumn::Ints(a, _), ExprColumn::Ints(b, _)) => {
            Some(compare_i64_pairs(a, b, op, &nulls))
        }
        (ExprColumn::Floats(a, _), ExprColumn::Floats(b, _)) => {
            Some(compare_f64_pairs(a, b, op, &nulls))
        }
        _ => None,
    };

    if let Some(mask) = mask {
        // A typed lane's operands are numeric on both sides, so the only way a
        // comparison is null is a null operand — already in `nulls`.
        return ExprColumn::Bools(mask, nulls);
    }

    // Generic lane. A constant right-hand side keeps the kernel's shape.
    if let ExprColumn::Scalar(constant) = rhs
        && let ExprColumn::Values(data) = lhs
    {
        let (mask, nulls) = compare_value_column_3vl(data, op, constant);
        return ExprColumn::Bools(mask, nulls);
    }

    let mut mask = vec![false; len];
    let mut nulls = NullBitmap::none(len);
    for (i, slot) in mask.iter_mut().enumerate() {
        match compare_values(&lhs.get(i), &rhs.get(i), op) {
            Some(pass) => *slot = pass,
            None => nulls.set(i),
        }
    }
    ExprColumn::Bools(mask, nulls)
}

#[allow(clippy::needless_range_loop)]
fn compare_i64_pairs(
    a: &[i64],
    b: &[i64],
    op: CmpOp,
    nulls: &NullBitmap,
) -> Vec<bool> {
    let len = a.len();
    let mut out = vec![false; len];
    for i in 0..len {
        out[i] = match op {
            CmpOp::Eq => a[i] == b[i],
            CmpOp::Neq => a[i] != b[i],
            CmpOp::Lt => a[i] < b[i],
            CmpOp::Le => a[i] <= b[i],
            CmpOp::Gt => a[i] > b[i],
            CmpOp::Ge => a[i] >= b[i],
        };
    }
    mask_out_nulls(&mut out, nulls);
    out
}

#[allow(clippy::needless_range_loop)]
fn compare_f64_pairs(
    a: &[f64],
    b: &[f64],
    op: CmpOp,
    nulls: &NullBitmap,
) -> Vec<bool> {
    let len = a.len();
    let mut out = vec![false; len];
    for i in 0..len {
        out[i] = match op {
            CmpOp::Eq => a[i] == b[i],
            CmpOp::Neq => a[i] != b[i],
            CmpOp::Lt => a[i] < b[i],
            CmpOp::Le => a[i] <= b[i],
            CmpOp::Gt => a[i] > b[i],
            CmpOp::Ge => a[i] >= b[i],
        };
    }
    mask_out_nulls(&mut out, nulls);
    out
}

fn mask_out_nulls(
    mask: &mut [bool],
    nulls: &NullBitmap,
) {
    if nulls.any_null() {
        for (i, slot) in mask.iter_mut().enumerate() {
            if nulls.is_null(i) {
                *slot = false;
            }
        }
    }
}

/// `NOT`, three-valued.
fn negate_bools(
    child: &ExprColumn,
    len: usize,
) -> Result<ExprColumn, String> {
    let mut mask = vec![false; len];
    let mut nulls = NullBitmap::none(len);
    for (i, slot) in mask.iter_mut().enumerate() {
        match child.get(i) {
            Value::Bool(b) => *slot = !b,
            Value::Null => nulls.set(i),
            v => {
                return Err(format!(
                    "Type mismatch: expected Boolean or Null but was {}",
                    v.name()
                ));
            }
        }
    }
    Ok(ExprColumn::Bools(mask, nulls))
}

/// Arithmetic, with typed lanes for the numeric shapes and the `Value`
/// operator itself for everything else (string concat, list append, temporal
/// arithmetic, type errors).
fn arithmetic(
    op: &ExprIR<Variable>,
    lhs: ExprColumn,
    rhs: ExprColumn,
    len: usize,
) -> Result<ExprColumn, String> {
    let nulls = union_nulls(&lhs, &rhs, len);
    // Int lanes: identical to `Value`'s wrapping arithmetic. Div/Modulo keep
    // the divide-by-zero error, so they only take the lane when no divisor is
    // zero — checking is one pass and the error path is rare.
    if let (Some(a), Some(b)) = (int_lane(&lhs, len), int_lane(&rhs, len)) {
        let out: Option<Vec<i64>> = match op {
            ExprIR::Add => Some((0..len).map(|i| a[i].wrapping_add(b[i])).collect()),
            ExprIR::Sub => Some((0..len).map(|i| a[i].wrapping_sub(b[i])).collect()),
            ExprIR::Mul => Some((0..len).map(|i| a[i].wrapping_mul(b[i])).collect()),
            ExprIR::Div | ExprIR::Modulo if (0..len).all(|i| b[i] != 0 || nulls.is_null(i)) => {
                let div = matches!(op, ExprIR::Div);
                Some(
                    (0..len)
                        .map(|i| {
                            if nulls.is_null(i) {
                                0
                            } else if div {
                                a[i].wrapping_div(b[i])
                            } else {
                                a[i].wrapping_rem(b[i])
                            }
                        })
                        .collect(),
                )
            }
            _ => None,
        };
        if let Some(data) = out {
            return Ok(ExprColumn::Ints(data, nulls));
        }
    }

    // Float lane, including the `Int`-against-`Float` promotion `Value`
    // performs itself. At least one operand must actually be a float, or
    // `Int / Int` would stop being integer division.
    if (float_operand(&lhs) || float_operand(&rhs))
        && let (Some(a), Some(b)) = (float_lane(&lhs, len), float_lane(&rhs, len))
    {
        let out: Option<Vec<f64>> = match op {
            ExprIR::Add => Some((0..len).map(|i| a[i] + b[i]).collect()),
            ExprIR::Sub => Some((0..len).map(|i| a[i] - b[i]).collect()),
            ExprIR::Mul => Some((0..len).map(|i| a[i] * b[i]).collect()),
            ExprIR::Div => Some((0..len).map(|i| a[i] / b[i]).collect()),
            ExprIR::Modulo => Some((0..len).map(|i| a[i] % b[i]).collect()),
            _ => None,
        };
        if let Some(data) = out {
            return Ok(ExprColumn::Floats(data, nulls));
        }
    }

    // Generic lane: the `Value` operator per element, so semantics and error
    // messages are the per-row evaluator's by construction.
    let lhs = lhs.into_values(len);
    let rhs = rhs.into_values(len);
    let mut out = Vec::with_capacity(len);
    for (a, b) in lhs.into_iter().zip(rhs) {
        out.push(match op {
            ExprIR::Add => (a + b)?,
            ExprIR::Sub => (a - b)?,
            ExprIR::Mul => (a * b)?,
            ExprIR::Div => (a / b)?,
            _ => (a % b)?,
        });
    }
    Ok(ExprColumn::Values(out))
}

/// The operand as `i64`s when every non-null row is an `Int`.
fn int_lane(
    column: &ExprColumn,
    len: usize,
) -> Option<Vec<i64>> {
    match column {
        ExprColumn::Ints(data, _) => Some(data.clone()),
        ExprColumn::Scalar(Value::Int(v)) => Some(vec![*v; len]),
        _ => None,
    }
}

/// The operand as `f64`s when every non-null row is numeric.
fn float_lane(
    column: &ExprColumn,
    len: usize,
) -> Option<Vec<f64>> {
    match column {
        ExprColumn::Floats(data, _) => Some(data.clone()),
        ExprColumn::Ints(data, _) => Some(data.iter().map(|&i| i as f64).collect()),
        ExprColumn::Scalar(Value::Float(v)) => Some(vec![*v; len]),
        ExprColumn::Scalar(Value::Int(v)) => Some(vec![*v as f64; len]),
        _ => None,
    }
}

/// Whether this operand contributes a float, which decides whether the float
/// lane applies at all (`Int / Int` must stay integer division).
const fn float_operand(column: &ExprColumn) -> bool {
    matches!(
        column,
        ExprColumn::Floats(..) | ExprColumn::Scalar(Value::Float(_))
    )
}
