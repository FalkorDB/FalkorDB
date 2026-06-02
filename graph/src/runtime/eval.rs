//! Standalone expression evaluator.
//!
//! [`ExprEval`] encapsulates all expression evaluation logic. It is used by
//! both the runtime (via `ExprEval::from_runtime`) and the optimizer (via
//! `ExprEval::constant()` for compile-time constant folding).
//!
//! ```text
//!  Expression tree (ExprIR)          Evaluation
//!  ========================          ==========
//!
//!        Add                     eval(Add)
//!       /   \                       |
//!    Mul     Int(1)              eval(Mul) + eval(Int(1))
//!   /   \                          |
//! Var(a) Var(b)               env[a] * env[b]  +  1
//! ```
//!
//! ## Two Evaluation Modes
//!
//! - **Runtime mode** (`ExprEval::from_runtime`): Full evaluation with access
//!   to the graph, environment bindings, parameters, and function calls.
//!   Used during query execution.
//!
//! - **Constant mode** (`ExprEval::constant`): No graph, no env, no functions.
//!   Used by the optimizer to fold constant sub-expressions at plan time.
//!   Returns `Err` if the expression references variables or functions.
//!
//! ## Evaluation Strategy
//!
//! The evaluator uses a hybrid approach:
//! - **Leaf nodes** (Null, Bool, Int, Float, String, Variable, Parameter, Map)
//!   are handled via fast-path early returns at the top of `eval()`.
//! - **Compound nodes** (Add, Sub, Mul, And, Or, List, etc.) use a stack-based
//!   iterative loop to avoid deep recursion. Children are pushed onto a stack
//!   and results accumulate in a `Vec<Value>`.
//! - **Complex nodes** (Quantifier, ListComprehension, Reduce, ShortestPath)
//!   fall back to recursive `eval()` calls since they need scoped env mutations.
//!
//! ## ValueIter
//!
//! [`ValueIter`] is a lazy iterator over values, used by `UNWIND` and list
//! comprehensions. It optimizes `range()` calls by producing integers on the
//! fly without materializing the entire list.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::Arc;

use orx_tree::{Dyn, DynNode, DynTree, NodeIdx, NodeRef};
use thin_vec::{ThinVec, thin_vec};

use crate::{
    parser::ast::{ExprIR, QuantifierType, Variable},
    runtime::{
        env::Env,
        functions::{FnType, apply_pow},
        ordermap::OrderMap,
        pool::Pool,
        runtime::Runtime,
        value::{CompareValue, Contains, DisjointOrNull, Value},
    },
};

// ---------------------------------------------------------------------------
// ValueIter
// ---------------------------------------------------------------------------

pub enum ValueIter {
    Empty,
    Once(Option<Value>),
    RangeUp { current: i64, end: i64, step: usize },
    RangeDown { current: i64, end: i64, step: usize },
    List(thin_vec::IntoIter<Value>),
}

impl Iterator for ValueIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Once(v) => v.take(),
            Self::RangeUp { current, end, step } => {
                if *current > *end {
                    return None;
                }
                let val = *current;
                *current += *step as i64;
                Some(Value::Int(val))
            }
            Self::RangeDown { current, end, step } => {
                if *current < *end {
                    return None;
                }
                let val = *current;
                *current -= *step as i64;
                Some(Value::Int(val))
            }
            Self::List(iter) => iter.next(),
        }
    }
}

// ---------------------------------------------------------------------------
// ExprEval
// ---------------------------------------------------------------------------

/// Shared expression evaluator used by both the runtime and the optimizer.
pub struct ExprEval<'a> {
    /// Full runtime context. `None` when evaluating constant expressions at
    /// plan time (optimizer).
    runtime: Option<&'a Runtime<'a>>,
    /// Value pool for reusable stack buffers and env cloning. `None` in
    /// constant-evaluation mode.
    pool: Option<&'a Pool<Value>>,
}

impl<'a> ExprEval<'a> {
    /// Full evaluation context backed by a [`Runtime`].
    pub const fn from_runtime(rt: &'a Runtime<'a>) -> Self {
        Self {
            runtime: Some(rt),
            pool: Some(rt.env_pool),
        }
    }

    /// Constant-only evaluation — no graph, no env, no functions.
    /// Any non-constant branch returns `Err`.
    #[must_use]
    pub const fn constant() -> Self {
        Self {
            runtime: None,
            pool: None,
        }
    }

    /// Convenience: unwrap the runtime or return a descriptive error.
    fn rt(&self) -> Result<&'a Runtime<'a>, String> {
        self.runtime
            .ok_or_else(|| String::from("not a constant expression"))
    }

    /// Clone an environment using the pool (required for Quantifier /
    /// ListComprehension).
    fn clone_env<'b>(
        &self,
        env: &Env<'b>,
    ) -> Result<Env<'b>, String>
    where
        'a: 'b,
    {
        let pool = self
            .pool
            .ok_or_else(|| String::from("not a constant expression"))?;
        Ok(env.clone_pooled(pool))
    }

    /// Resolve an environment variable.
    fn resolve_var(
        env: Option<&Env<'_>>,
        x: &Variable,
    ) -> Result<Value, String> {
        env.and_then(|e| e.get(x))
            .ok_or_else(|| format!("Variable {} not found", x.as_str()))
            .cloned()
    }

    // -------------------------------------------------------------------
    // Main evaluator
    // -------------------------------------------------------------------

    pub fn eval(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: Option<&Env<'_>>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        // Fast-path early returns for leaf / simple nodes.
        match ir.node(idx).data() {
            ExprIR::Constant(v) => return Ok(v.clone()),
            ExprIR::Variable(x) => {
                return Self::resolve_var(env, x);
            }
            ExprIR::Parameter(x) => {
                let rt = self.rt()?;
                return rt.parameters.get(x).map_or_else(
                    || Err(format!("Parameter {x} not found")),
                    |v| Ok(v.clone()),
                );
            }
            ExprIR::Map => {
                return Ok(Value::Map(Arc::new(
                    ir.node(idx)
                        .children()
                        .map(|child| {
                            Ok((
                                if let ExprIR::Constant(Value::String(key)) = child.data() {
                                    key.clone()
                                } else {
                                    return Err("Map key must be a string".into());
                                },
                                self.eval(ir, child.child(0).idx(), env, agg_group_key)?,
                            ))
                        })
                        .collect::<Result<_, String>>()?,
                )));
            }
            ExprIR::MapProjection => {
                return self.eval_map_projection(ir, idx, env, agg_group_key);
            }
            ExprIR::ShortestPath {
                rel_types,
                min_hops,
                max_hops,
                directed,
                all_paths,
            } => {
                return self.eval_shortest_path(
                    ir,
                    idx,
                    env,
                    agg_group_key,
                    rel_types,
                    *min_hops,
                    *max_hops,
                    *directed,
                    *all_paths,
                );
            }
            _ => {}
        }

        // Stack-based iterative evaluation. When a pool is available, use a
        // Pooled handle so early-return paths (errors, breaks) recycle the
        // buffer via Pooled's Drop. Constant-only eval has no pool — fall
        // back to a heap Vec.
        let mut res_pooled = self.pool.map(|pool| pool.acquire(0));
        let mut res_owned: Vec<Value> = Vec::new();
        let res: &mut Vec<Value> = match res_pooled.as_mut() {
            Some(p) => &mut *p,
            None => &mut res_owned,
        };
        res.clear();

        let mut stack = thin_vec![(idx, false)];
        while let Some((idx, reenter)) = stack.pop() {
            let node = ir.node(idx);
            match node.data() {
                ExprIR::Constant(v) => res.push(v.clone()),
                ExprIR::Variable(x) => res.push(Self::resolve_var(env, x)?),
                ExprIR::Parameter(x) => res.push(self.rt()?.parameters.get(x).map_or_else(
                    || Err(format!("Parameter {x} not found")),
                    |v| Ok(v.clone()),
                )?),
                ExprIR::List => {
                    if reenter {
                        let mut list = thin_vec![];
                        for _ in 0..node.num_children() {
                            list.push(res.pop().unwrap());
                        }
                        res.push(Value::List(Arc::new(list)));
                    } else if node.num_children() > 0 {
                        stack.push((idx, true));
                        for idx in node.children().map(|c| c.idx()) {
                            stack.push((idx, false));
                        }
                    } else {
                        res.push(Value::List(Arc::new(thin_vec![])));
                    }
                }
                ExprIR::Length => match self.eval(ir, node.child(0).idx(), env, agg_group_key)? {
                    Value::List(arr) => res.push(Value::Int(arr.len() as _)),
                    _ => return Err(String::from("Length operator requires a list")),
                },
                ExprIR::GetElement => {
                    let arr = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    let i = self.eval(ir, node.child(1).idx(), env, agg_group_key)?;
                    match (arr, i) {
                        (Value::List(values), Value::Int(i)) => {
                            let len = values.len() as i64;
                            let normalized_index = if i < 0 { len + i } else { i };
                            if normalized_index >= 0 && normalized_index < len {
                                res.push(values[normalized_index as usize].clone());
                            } else {
                                res.push(Value::Null);
                            }
                        }
                        (Value::List(_), Value::Null) => {
                            res.push(Value::Null);
                        }
                        (Value::List(_), v) => {
                            return Err(format!(
                                "Type mismatch: expected Integer but was {}",
                                v.name()
                            ));
                        }
                        (Value::Node(id), Value::String(key)) => {
                            let rt = self.rt()?;
                            res.push(rt.get_node_attribute(id, &key).unwrap_or(Value::Null));
                        }
                        (Value::Relationship(rel), Value::String(key)) => {
                            let rt = self.rt()?;
                            res.push(
                                rt.get_relationship_attribute(rel.0, &key)
                                    .unwrap_or(Value::Null),
                            );
                        }
                        (Value::Map(map), Value::String(key)) => {
                            res.push(map.get(&key).map_or(Value::Null, std::clone::Clone::clone));
                        }
                        (Value::Map(_), Value::Null) | (Value::Null, _) => res.push(Value::Null),
                        v => return Err(format!("Type mismatch: unexpected types {v:?}")),
                    }
                }
                ExprIR::GetElements => {
                    let arr = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    let a = self.eval(ir, node.child(1).idx(), env, agg_group_key)?;
                    let b = self.eval(ir, node.child(2).idx(), env, agg_group_key)?;
                    res.push(get_elements(&arr, &a, &b)?);
                }
                ExprIR::IsNode => match self.eval(ir, node.child(0).idx(), env, agg_group_key)? {
                    Value::Node(_) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::IsRelationship => {
                    match self.eval(ir, node.child(0).idx(), env, agg_group_key)? {
                        Value::Relationship(_) => res.push(Value::Bool(true)),
                        _ => res.push(Value::Bool(false)),
                    }
                }
                ExprIR::Or => {
                    let mut is_null = false;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval(ir, child.idx(), env, agg_group_key)? {
                            Value::Bool(true) => {
                                found = true;
                                res.push(Value::Bool(true));
                                break;
                            }
                            Value::Bool(false) => {}
                            Value::Null => is_null = true,
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        if is_null {
                            res.push(Value::Null);
                        } else {
                            res.push(Value::Bool(false));
                        }
                    }
                }
                ExprIR::Xor => {
                    let mut last = None;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval(ir, child.idx(), env, agg_group_key)? {
                            Value::Bool(b) => last = Some(last.map_or(b, |l| logical_xor(l, b))),
                            Value::Null => {
                                found = true;
                                res.push(Value::Null);
                                break;
                            }
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        res.push(Value::Bool(last.unwrap_or(false)));
                    }
                }
                ExprIR::And => {
                    let mut is_null = false;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval(ir, child.idx(), env, agg_group_key)? {
                            Value::Bool(false) => {
                                found = true;
                                res.push(Value::Bool(false));
                                break;
                            }
                            Value::Bool(true) => {}
                            Value::Null => is_null = true,
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        if is_null {
                            res.push(Value::Null);
                        } else {
                            res.push(Value::Bool(true));
                        }
                    }
                }
                ExprIR::Not => match self.eval(ir, node.child(0).idx(), env, agg_group_key)? {
                    Value::Bool(b) => res.push(Value::Bool(!b)),
                    Value::Null => res.push(Value::Null),
                    v => {
                        return Err(format!(
                            "Type mismatch: expected Boolean or Null but was {}",
                            v.name()
                        ));
                    }
                },
                ExprIR::Negate => match self.eval(ir, node.child(0).idx(), env, agg_group_key)? {
                    Value::Int(i) => res.push(Value::Int(i.checked_neg().ok_or_else(|| {
                        String::from("ArgumentError: integer overflow in unary minus")
                    })?)),
                    Value::Float(f) => res.push(Value::Float(-f)),
                    Value::Null => res.push(Value::Null),
                    v => {
                        return Err(format!(
                            "Type mismatch: expected Integer, Float, or Null but was {}",
                            v.name()
                        ));
                    }
                },
                ExprIR::Eq => res.push(all_equals(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key)),
                )?),
                ExprIR::Neq => res.push(all_not_equals(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key)),
                )?),
                ExprIR::Lt => match self
                    .eval(ir, node.child(0).idx(), env, agg_group_key)?
                    .compare_value(&self.eval(ir, node.child(1).idx(), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Less, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Gt => match self
                    .eval(ir, node.child(0).idx(), env, agg_group_key)?
                    .compare_value(&self.eval(ir, node.child(1).idx(), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Greater, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Le => match self
                    .eval(ir, node.child(0).idx(), env, agg_group_key)?
                    .compare_value(&self.eval(ir, node.child(1).idx(), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Less | Ordering::Equal, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Ge => match self
                    .eval(ir, node.child(0).idx(), env, agg_group_key)?
                    .compare_value(&self.eval(ir, node.child(1).idx(), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Greater | Ordering::Equal, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::In => {
                    let value = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    let list = self.eval(ir, node.child(1).idx(), env, agg_group_key)?;
                    res.push(list_contains(&list, value)?);
                }
                ExprIR::Add => res.push(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(|acc, value| acc? + value?)
                        .ok_or_else(|| {
                            String::from("Add operator requires at least one operand")
                        })??,
                ),
                ExprIR::Sub => res.push(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(|acc, value| acc? - value?)
                        .ok_or_else(|| {
                            String::from("Sub operator requires at least one argument")
                        })??,
                ),
                ExprIR::Mul => res.push(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(|acc, value| acc? * value?)
                        .ok_or_else(|| {
                            String::from("Mul operator requires at least one argument")
                        })??,
                ),
                ExprIR::Div => res.push(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(|acc, value| acc? / value?)
                        .ok_or_else(|| {
                            String::from("Div operator requires at least one argument")
                        })??,
                ),
                ExprIR::Modulo => res.push(
                    node.children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(|acc, value| acc? % value?)
                        .ok_or_else(|| {
                            String::from("Modulo operator requires at least one argument")
                        })??,
                ),
                ExprIR::Pow => res.push(
                    node.children()
                        .flat_map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .reduce(apply_pow)
                        .ok_or_else(|| {
                            String::from("Pow operator requires at least one argument")
                        })?,
                ),
                ExprIR::Distinct => {
                    let rt = self.rt()?;
                    let group_id = agg_group_key.unwrap();
                    let values = node
                        .children()
                        .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                        .collect::<Result<ThinVec<_>, _>>()?;
                    let mut value_dedupers = rt.value_dedupers.borrow_mut();
                    let value_deduper = value_dedupers.entry((idx, group_id)).or_default();
                    if value_deduper.is_seen(&values) {
                        res.push(Value::List(Arc::new(thin_vec![Value::Null])));
                    } else {
                        res.push(Value::List(Arc::new(values)));
                    }
                }
                ExprIR::Property(attr) => {
                    let obj = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    match obj {
                        Value::Node(id) => {
                            let rt = self.rt()?;
                            res.push(rt.get_node_attribute(id, attr).unwrap_or(Value::Null));
                        }
                        Value::Relationship(rel) => {
                            let rt = self.rt()?;
                            res.push(
                                rt.get_relationship_attribute(rel.0, attr)
                                    .unwrap_or(Value::Null),
                            );
                        }
                        other => {
                            res.push(other.get_attr(attr)?);
                        }
                    }
                }
                ExprIR::FuncInvocation(func) => {
                    let rt = self.rt()?;
                    if agg_group_key.is_none()
                        && let FnType::Aggregation {
                            finalizer: finalize,
                            ..
                        } = &func.fn_type
                        && let ExprIR::Variable(key) = node.child(node.num_children() - 1).data()
                    {
                        let e = env.ok_or_else(|| String::from("Variable not found"))?;
                        let acc = e.get(key).unwrap().clone();

                        return match finalize {
                            Some(func) => Ok((func)(acc)),
                            None => Ok(acc),
                        };
                    }
                    // Fast path for struct-constructor functions (duration,
                    // date, point, ...): when the binder rewrote a
                    // `{key: value}` map call into positional children, write
                    // child values into a stack array and dispatch the
                    // slice-based struct_fn directly. Bypasses the per-row
                    // ThinVec<Value> heap alloc, the validate_args_type walk,
                    // and the Arc<RuntimeFn> dispatch hop. The
                    // `num_children > 1` guard distinguishes the rewritten
                    // positional form from the regular `duration({...})` call
                    // with a single Map argument. Max 10 slots
                    // (matches `localdatetime`).
                    if let Some(struct_fn) = func.struct_fn
                        && node.num_children() > 1
                    {
                        let n = node.num_children();
                        debug_assert!(n <= 10);
                        let mut slots: [Value; 10] = std::array::from_fn(|_| Value::Null);
                        for (i, child) in node.children().enumerate() {
                            if !matches!(child.data(), ExprIR::Constant(Value::Null)) {
                                slots[i] = self.eval(ir, child.idx(), env, agg_group_key)?;
                            }
                        }
                        res.push(struct_fn(&slots[..n])?);
                        continue;
                    }
                    // Distinct-prefixed aggregate (e.g. `count(DISTINCT x, acc)`):
                    // the first child carries the already-deduplicated list;
                    // flatten it with the accumulator. Needs owned ownership of
                    // the inner list to avoid a copy, so it stays on the
                    // ThinVec path.
                    if node.num_children() == 2 && matches!(node.child(0).data(), ExprIR::Distinct)
                    {
                        let mut args = node
                            .children()
                            .map(|child| self.eval(ir, child.idx(), env, agg_group_key))
                            .collect::<Result<ThinVec<_>, _>>()?;
                        match args.remove(0) {
                            Value::List(values) => {
                                let mut values = Arc::unwrap_or_clone(values);
                                values.append(&mut args);
                                args = values;
                            }
                            _ => unreachable!(),
                        }
                        func.validate_args_type(&args)?;
                        if !rt.write && func.write {
                            return Err(String::from(
                                "graph.RO_QUERY is to be executed only on read-only queries",
                            ));
                        }
                        res.push(func.func.call(rt, &args)?);
                        continue;
                    }

                    // Common path: evaluate children directly onto the shared
                    // `res` stack and pass a slice to the function. Eliminates
                    // the per-call ThinVec<Value> heap allocation that an
                    // intermediate `args` collection would do — same trick the
                    // struct_fn fast path above uses with a stack array.
                    let base = res.len();
                    for child in node.children() {
                        let v = self.eval(ir, child.idx(), env, agg_group_key)?;
                        res.push(v);
                    }
                    func.validate_args_type(&res[base..])?;
                    if !rt.write && func.write {
                        return Err(String::from(
                            "graph.RO_QUERY is to be executed only on read-only queries",
                        ));
                    }
                    let out = func.func.call(rt, &res[base..])?;
                    res.truncate(base);
                    res.push(out);
                }
                ExprIR::Map => res.push(Value::Map(Arc::new(
                    node.children()
                        .map(|child| {
                            Ok((
                                if let ExprIR::Constant(Value::String(key)) = child.data() {
                                    key.clone()
                                } else {
                                    return Err("Map key must be a string".into());
                                },
                                self.eval(ir, child.child(0).idx(), env, agg_group_key)?,
                            ))
                        })
                        .collect::<Result<_, String>>()?,
                ))),
                ExprIR::MapProjection => {
                    res.push(self.eval_map_projection(ir, idx, env, agg_group_key)?);
                }
                ExprIR::Quantifier {
                    quantifier_type: quantifier,
                    var,
                } => {
                    let list = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    match list {
                        Value::List(values) => {
                            let e = env.ok_or_else(|| String::from("Variable not found"))?;
                            let mut env = self.clone_env(e)?;
                            let mut t = 0;
                            let mut f = 0;
                            let mut n = 0;
                            for value in values.iter().cloned() {
                                env.insert(var, value);

                                match self.eval(
                                    ir,
                                    node.child(1).idx(),
                                    Some(&env),
                                    agg_group_key,
                                )? {
                                    Value::Bool(true) => t += 1,
                                    Value::Bool(false) => f += 1,
                                    Value::Null => n += 1,
                                    value => {
                                        return Err(format!(
                                            "Type mismatch: expected Boolean but was {}",
                                            value.name()
                                        ));
                                    }
                                }
                            }

                            res.push(eval_quantifier(quantifier, t, f, n));
                        }
                        Value::Null => res.push(Value::Null),
                        value => {
                            return Err(format!(
                                "Type mismatch: expected List but was {}",
                                value.name()
                            ));
                        }
                    }
                }
                ExprIR::ListComprehension(var) => {
                    let e = env.ok_or_else(|| String::from("Variable not found"))?;
                    let iter = self.eval_iter_expr(ir, node.child(0).idx(), env)?;
                    let mut env = self.clone_env(e)?;
                    let mut acc = thin_vec![];
                    for value in iter {
                        env.insert(var, value);
                        match self.eval(ir, node.child(1).idx(), Some(&env), agg_group_key)? {
                            Value::Bool(true) => {}
                            _ => continue,
                        }
                        acc.push(self.eval(ir, node.child(2).idx(), Some(&env), agg_group_key)?);
                    }

                    res.push(Value::List(Arc::new(acc)));
                }
                ExprIR::Reduce {
                    accumulator: acc_var,
                    iterator: iter_var,
                } => {
                    // child[0] = init, child[1] = list, child[2] = body
                    let init = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
                    let list = self.eval(ir, node.child(1).idx(), env, agg_group_key)?;
                    match list {
                        Value::List(values) => {
                            let e = env.ok_or_else(|| String::from("Variable not found"))?;
                            let mut env = self.clone_env(e)?;
                            let mut accumulator = init;
                            for value in values.iter().cloned() {
                                env.insert(acc_var, accumulator);
                                env.insert(iter_var, value);
                                accumulator =
                                    self.eval(ir, node.child(2).idx(), Some(&env), agg_group_key)?;
                            }
                            res.push(accumulator);
                        }
                        Value::Null => res.push(Value::Null),
                        value => {
                            return Err(format!(
                                "Type mismatch: expected List but was {}",
                                value.name()
                            ));
                        }
                    }
                }
                ExprIR::PatternComprehension(_) => {
                    unreachable!("PatternComprehension should be handled by the planner")
                }
                ExprIR::Paren => {
                    res.push(self.eval(ir, node.child(0).idx(), env, agg_group_key)?);
                }
                ExprIR::Pattern(_) => {
                    unreachable!("Pattern should be handled by the planner")
                }
                ExprIR::ShortestPath { .. } => {
                    unreachable!("ShortestPath should be handled in the early-return section")
                }
            }
        }
        debug_assert_eq!(res.len(), 1);
        let result = res.pop().unwrap();
        Ok(result)
    }

    // -------------------------------------------------------------------
    // Companion methods
    // -------------------------------------------------------------------

    pub fn eval_iter_expr(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: Option<&Env<'_>>,
    ) -> Result<ValueIter, String> {
        match ir.node(idx).data() {
            ExprIR::FuncInvocation(func) if func.name == "range" => {
                let start = self.eval(ir, ir.node(idx).child(0).idx(), env, None)?;
                let end = self.eval(ir, ir.node(idx).child(1).idx(), env, None)?;
                let step = ir
                    .node(idx)
                    .get_child(2)
                    .map_or_else(|| Ok(Value::Int(1)), |c| self.eval(ir, c.idx(), env, None))?;
                func.validate_args_type(&[&start, &end, &step])?;
                match (start, end, step) {
                    (Value::Int(start), Value::Int(end), Value::Int(step)) => {
                        if step == 0 {
                            return Err(String::from(
                                "ArgumentError: step argument to range() can't be 0",
                            ));
                        }
                        if (start > end && step > 0) || (start < end && step < 0) {
                            return Ok(ValueIter::Empty);
                        }
                        let span = if end >= start {
                            (end as i128) - (start as i128)
                        } else {
                            (start as i128) - (end as i128)
                        };
                        let abs_step = (step as i128).unsigned_abs();
                        let length = (span.unsigned_abs() / abs_step)
                            .checked_add(1)
                            .ok_or_else(|| String::from("Range too large"))?;
                        if length > u32::MAX as u128 {
                            return Err(String::from("Range too large"));
                        }

                        if step > 0 {
                            return Ok(ValueIter::RangeUp {
                                current: start,
                                end,
                                step: step as usize,
                            });
                        }
                        Ok(ValueIter::RangeDown {
                            current: start,
                            end,
                            step: step.unsigned_abs() as usize,
                        })
                    }
                    _ => {
                        unreachable!();
                    }
                }
            }
            _ => {
                let res = self.eval(ir, idx, env, None)?;
                match res {
                    Value::List(arr) => Ok(ValueIter::List(Arc::unwrap_or_clone(arr).into_iter())),
                    Value::Null => Ok(ValueIter::Empty),
                    _ => Ok(ValueIter::Once(Some(res))),
                }
            }
        }
    }

    /// Evaluate a `shortestPath()` or `allShortestPaths()` expression.
    ///
    /// Children: [source_var_expr, dest_var_expr]
    /// Returns a `Path` value (alternating nodes and edges) or `Null`.
    #[allow(clippy::too_many_arguments)]
    fn eval_shortest_path(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: Option<&Env<'_>>,
        agg_group_key: Option<u64>,
        rel_types: &[Arc<String>],
        min_hops: u32,
        max_hops: Option<u32>,
        directed: bool,
        all_paths: bool,
    ) -> Result<Value, String> {
        let node = ir.node(idx);
        let src_val = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;
        let dst_val = self.eval(ir, node.child(1).idx(), env, agg_group_key)?;

        let src_id = match &src_val {
            Value::Node(id) => *id,
            Value::Null => return Ok(Value::Null),
            _ => return Err("A shortestPath requires bound nodes".into()),
        };
        let dst_id = match &dst_val {
            Value::Node(id) => *id,
            Value::Null => return Ok(Value::Null),
            _ => return Err("A shortestPath requires bound nodes".into()),
        };

        let rt = self.rt()?;
        let g = rt.g.borrow();

        // min_hops == 0: if src == dest, return single-node path
        if min_hops == 0 && src_id == dst_id {
            let path: ThinVec<Value> = thin_vec![Value::Node(src_id)];
            if all_paths {
                return Ok(Value::List(Arc::new(thin_vec![Value::Path(Arc::new(
                    path
                ))])));
            }
            return Ok(Value::Path(Arc::new(path)));
        }

        // Build adjacency matrix filtered by rel_types
        let adj = g.build_adjacency_matrix(rel_types);

        // Also build transpose if undirected
        let adj_t = if directed {
            None
        } else {
            use crate::graph::graphblas::matrix::Transpose;
            Some(adj.transpose())
        };

        let max_level = max_hops.map_or(u64::MAX, |m| m as u64);
        let node_cap = g.node_cap();

        // Build adjacency list from the sparse matrix for efficient BFS.
        // Use a hash map keyed by source node so memory scales with the
        // number of edges actually present — not with `node_cap`, which
        // reflects allocated matrix capacity and can be much larger than
        // the live node count (SEC-3).
        let mut adj_list: rustc_hash::FxHashMap<u64, Vec<u64>> = rustc_hash::FxHashMap::default();
        for (row, col) in adj.iter(0, node_cap.saturating_sub(1)) {
            adj_list.entry(row).or_default().push(col);
        }
        if let Some(ref t) = adj_t {
            for (row, col) in t.iter(0, node_cap.saturating_sub(1)) {
                adj_list.entry(row).or_default().push(col);
            }
            // In undirected mode reciprocal edges (a→b and b→a) cause the
            // same neighbour to be inserted twice, which would double-count
            // predecessors during BFS and produce duplicate shortest paths.
            for neighbours in adj_list.values_mut() {
                neighbours.sort_unstable();
                neighbours.dedup();
            }
        }

        if all_paths {
            // All shortest paths: BFS to find distance, then enumerate
            Ok(self.bfs_all_shortest_paths(
                &g, &adj_list, src_id, dst_id, max_level, node_cap, rel_types, min_hops,
            ))
        } else {
            // Single shortest path via BFS with parent tracking
            Ok(self.bfs_shortest_path(
                &g, &adj_list, src_id, dst_id, max_level, node_cap, rel_types, min_hops,
            ))
        }
    }

    /// BFS to find the single shortest path between two nodes.
    #[allow(clippy::too_many_arguments)]
    fn bfs_shortest_path(
        &self,
        g: &crate::graph::graph::Graph,
        adj_list: &rustc_hash::FxHashMap<u64, Vec<u64>>,
        src_id: crate::graph::graph::NodeId,
        dst_id: crate::graph::graph::NodeId,
        max_level: u64,
        _node_cap: u64,
        rel_types: &[Arc<String>],
        min_hops: u32,
    ) -> Value {
        use crate::graph::graph::{NodeId, RelationshipId};

        let src = u64::from(src_id);
        let dst = u64::from(dst_id);

        // parent[n] = Some(prev) during BFS, keyed only by visited nodes
        // (SEC-3: bounded by visited count, not node_cap).
        let mut parent: rustc_hash::FxHashMap<u64, u64> = rustc_hash::FxHashMap::default();
        parent.insert(src, src); // mark source visited (self-parent)

        let mut queue: VecDeque<(u64, u64)> = VecDeque::new(); // (node, depth)
        queue.push_back((src, 0));

        let mut found = false;

        while let Some((cur, depth)) = queue.pop_front() {
            if depth >= max_level {
                continue;
            }
            if let Some(neighbours) = adj_list.get(&cur) {
                for &col in neighbours {
                    // Not using the Entry API: we only want to insert when
                    // absent and also `break` out of the loop on `dst`,
                    // which Entry::or_insert_with doesn't express cleanly.
                    #[allow(clippy::map_entry)]
                    if !parent.contains_key(&col) {
                        parent.insert(col, cur);
                        if col == dst {
                            found = true;
                            break;
                        }
                        queue.push_back((col, depth + 1));
                    }
                }
            }
            if found {
                break;
            }
        }

        if !found {
            return Value::Null;
        }

        // Reconstruct path from dst back to src
        let mut path_nodes: Vec<u64> = vec![dst];
        let mut cur = dst;
        while cur != src {
            cur = *parent.get(&cur).expect("BFS parent chain broken");
            path_nodes.push(cur);
        }
        path_nodes.reverse();

        // Enforce min_hops: path must have at least min_hops edges
        if (path_nodes.len() - 1) < min_hops as usize {
            return Value::Null;
        }

        // Build alternating node/relationship path
        let mut path: ThinVec<Value> = ThinVec::with_capacity(path_nodes.len() * 2 - 1);
        path.push(Value::Node(NodeId::from(path_nodes[0])));
        for i in 0..path_nodes.len() - 1 {
            let from = NodeId::from(path_nodes[i]);
            let to = NodeId::from(path_nodes[i + 1]);
            // Find the relationship between consecutive path nodes
            let rel_id: Option<(RelationshipId, NodeId, NodeId)> = g
                .get_src_dest_relationships(from, to, rel_types)
                .next()
                .map(|rid| (rid, from, to))
                .or_else(|| {
                    g.get_src_dest_relationships(to, from, rel_types)
                        .next()
                        .map(|rid| (rid, to, from))
                });
            if let Some((rid, src, dst)) = rel_id {
                path.push(Value::Relationship(Box::new((rid, src, dst))));
            }
            path.push(Value::Node(to));
        }

        Value::Path(Arc::new(path))
    }

    /// BFS to find all shortest paths between two nodes.
    #[allow(clippy::too_many_arguments)]
    fn bfs_all_shortest_paths(
        &self,
        g: &crate::graph::graph::Graph,
        adj_list: &rustc_hash::FxHashMap<u64, Vec<u64>>,
        src_id: crate::graph::graph::NodeId,
        dst_id: crate::graph::graph::NodeId,
        max_level: u64,
        _node_cap: u64,
        rel_types: &[Arc<String>],
        min_hops: u32,
    ) -> Value {
        use crate::graph::graph::{NodeId, RelationshipId};

        let src = u64::from(src_id);
        let dst = u64::from(dst_id);

        // BFS to find distance and all shortest-path predecessors.
        // Maps keyed by visited nodes only (SEC-3).
        let mut distances: rustc_hash::FxHashMap<u64, u64> = rustc_hash::FxHashMap::default();
        distances.insert(src, 0);

        // predecessors[n] = list of all nodes that are parents on some shortest path
        let mut predecessors: rustc_hash::FxHashMap<u64, Vec<u64>> =
            rustc_hash::FxHashMap::default();

        let mut queue: VecDeque<u64> = VecDeque::new();
        queue.push_back(src);

        let mut found_dist: Option<u64> = None;

        while let Some(cur) = queue.pop_front() {
            let cur_dist = *distances.get(&cur).expect("BFS dequeued unvisited node");
            if let Some(fd) = found_dist
                && cur_dist >= fd
            {
                continue;
            }
            if cur_dist >= max_level {
                continue;
            }
            let Some(neighbours) = adj_list.get(&cur) else {
                continue;
            };
            for &col in neighbours {
                let new_dist = cur_dist + 1;
                match distances.get(&col).copied() {
                    None => {
                        distances.insert(col, new_dist);
                        predecessors.entry(col).or_default().push(cur);
                        if col == dst {
                            found_dist = Some(new_dist);
                        } else {
                            queue.push_back(col);
                        }
                    }
                    Some(d) if d == new_dist => {
                        predecessors.entry(col).or_default().push(cur);
                    }
                    _ => {}
                }
            }
        }

        if found_dist.is_none() {
            return Value::Null;
        }

        // Enumerate all shortest paths by DFS from dst back to src
        let mut all_paths: Vec<ThinVec<Value>> = Vec::new();
        let mut stack: Vec<(u64, Vec<u64>)> = vec![(dst, vec![dst])];

        while let Some((cur, path_so_far)) = stack.pop() {
            if cur == src {
                // Reconstruct forward path
                let mut fwd = path_so_far.clone();
                fwd.reverse();
                let mut path: ThinVec<Value> = ThinVec::with_capacity(fwd.len() * 2 - 1);
                path.push(Value::Node(NodeId::from(fwd[0])));
                for i in 0..fwd.len() - 1 {
                    let from = NodeId::from(fwd[i]);
                    let to = NodeId::from(fwd[i + 1]);
                    let rel_id: Option<(RelationshipId, NodeId, NodeId)> = g
                        .get_src_dest_relationships(from, to, rel_types)
                        .next()
                        .map(|rid| (rid, from, to))
                        .or_else(|| {
                            g.get_src_dest_relationships(to, from, rel_types)
                                .next()
                                .map(|rid| (rid, to, from))
                        });
                    if let Some((rid, src, dst)) = rel_id {
                        path.push(Value::Relationship(Box::new((rid, src, dst))));
                    }
                    path.push(Value::Node(to));
                }
                all_paths.push(path);
                continue;
            }
            if let Some(preds) = predecessors.get(&cur) {
                for &pred in preds {
                    let mut new_path = path_so_far.clone();
                    new_path.push(pred);
                    stack.push((pred, new_path));
                }
            }
        }

        // Filter paths by min_hops and return list of paths
        let result: ThinVec<Value> = all_paths
            .into_iter()
            .filter(|p| {
                // Number of hops = number of nodes - 1; nodes are at even indices
                let node_count = p.iter().filter(|v| matches!(v, Value::Node(_))).count();
                node_count > 0 && (node_count - 1) >= min_hops as usize
            })
            .map(|p| Value::Path(Arc::new(p)))
            .collect();
        Value::List(Arc::new(result))
    }

    pub(crate) fn eval_map_projection(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: Option<&Env<'_>>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        let rt = self.rt()?;
        let node = ir.node(idx);
        let base = self.eval(ir, node.child(0).idx(), env, agg_group_key)?;

        if matches!(base, Value::Null) {
            return Ok(Value::Null);
        }

        if !matches!(
            &base,
            Value::Node(_) | Value::Relationship(_) | Value::Map(_)
        ) {
            return Err("Encountered unhandled type evaluating map projection".to_string());
        }

        let mut result = OrderMap::default();

        for i in 1..node.num_children() {
            let item = node.child(i);
            match item.data() {
                ExprIR::MapProjection => match &base {
                    Value::Node(id) => {
                        for (k, v) in rt.get_node_attrs(*id) {
                            result.insert(k, v);
                        }
                    }
                    Value::Relationship(rel) => {
                        for (k, v) in rt.get_relationship_attrs(rel.0) {
                            result.insert(k, v);
                        }
                    }
                    Value::Map(map) => {
                        for (k, v) in map.iter() {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                    _ => {
                        return Err(
                            "Encountered unhandled type evaluating map projection".to_string()
                        );
                    }
                },
                ExprIR::Property(prop_name) => {
                    let value = match &base {
                        Value::Node(id) => {
                            rt.get_node_attribute(*id, prop_name).unwrap_or(Value::Null)
                        }
                        Value::Relationship(rel) => rt
                            .get_relationship_attribute(rel.0, prop_name)
                            .unwrap_or(Value::Null),
                        Value::Map(map) => map.get(prop_name).cloned().unwrap_or(Value::Null),
                        _ => {
                            return Err(
                                "Encountered unhandled type evaluating map projection".to_string()
                            );
                        }
                    };
                    result.insert(prop_name.clone(), value);
                }
                ExprIR::Constant(Value::String(_)) => {
                    let key = if let ExprIR::Constant(Value::String(k)) = item.data() {
                        k.clone()
                    } else {
                        unreachable!();
                    };
                    let value = self.eval(ir, item.child(0).idx(), env, agg_group_key)?;
                    result.insert(key, value);
                }
                _ => {
                    return Err("Encountered unhandled type evaluating map projection".to_string());
                }
            }
        }

        Ok(Value::Map(Arc::new(result)))
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (no Runtime dependency)
// ---------------------------------------------------------------------------

pub(crate) const fn eval_quantifier(
    quantifier_type: &QuantifierType,
    true_count: usize,
    false_count: usize,
    null_count: usize,
) -> Value {
    match quantifier_type {
        QuantifierType::All => {
            if false_count > 0 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(true)
            }
        }
        QuantifierType::Any => {
            if true_count > 0 {
                Value::Bool(true)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(false)
            }
        }
        QuantifierType::None => {
            if true_count > 0 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(true)
            }
        }
        QuantifierType::Single => {
            if true_count == 1 && null_count == 0 {
                Value::Bool(true)
            } else if true_count > 1 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(false)
            }
        }
    }
}

// the semantic of Eq [1, 2, 3] is: 1 EQ 2 AND 2 EQ 3
pub(crate) fn all_equals<I>(mut iter: I) -> Result<Value, String>
where
    I: Iterator<Item = Result<Value, String>>,
{
    if let Some(first) = iter.next() {
        let prev = first?;
        for next in iter {
            let next = next?;
            match prev.compare_value(&next) {
                (_, DisjointOrNull::ComparedNull) => return Ok(Value::Null),
                (_, DisjointOrNull::NaN | DisjointOrNull::Disjoint) => {
                    return Ok(Value::Bool(false));
                }
                (Ordering::Equal, _) => {}
                _ => return Ok(Value::Bool(false)),
            }
        }
        Ok(Value::Bool(true))
    } else {
        Err(String::from("Eq operator requires at least two arguments"))
    }
}

pub(crate) fn all_not_equals<I>(mut iter: I) -> Result<Value, String>
where
    I: Iterator<Item = Result<Value, String>>,
{
    if let Some(first) = iter.next() {
        let prev = first?;
        for next in iter {
            let next = next?;
            match prev.partial_cmp(&next) {
                None => return Ok(Value::Null),
                Some(Ordering::Less | Ordering::Greater) => {}
                Some(Ordering::Equal) => return Ok(Value::Bool(false)),
            }
        }
        Ok(Value::Bool(true))
    } else {
        Err(String::from("Eq operator requires at least two arguments"))
    }
}

pub(crate) fn list_contains(
    list: &Value,
    value: Value,
) -> Result<Value, String> {
    match list {
        Value::List(l) => Ok(Contains::contains(l.as_ref(), value)),
        Value::Null => Ok(Value::Null),
        _ => Err(format!(
            "Type mismatch: expected List or Null but was {}",
            list.name()
        )),
    }
}

pub(crate) fn get_elements(
    arr: &Value,
    start: &Value,
    end: &Value,
) -> Result<Value, String> {
    match (arr, start, end) {
        (Value::List(values), Value::Int(start), Value::Int(end)) => {
            let mut start = *start;
            let mut end = *end;
            if start < 0 {
                start = (values.len() as i64 + start).max(0);
            }
            if end < 0 {
                end = (values.len() as i64 + end).max(0);
            } else {
                end = end.min(values.len() as i64);
            }
            if start > end {
                return Ok(Value::List(Arc::new(thin_vec![])));
            }
            Ok(Value::List(Arc::new(
                values[start as usize..end as usize]
                    .iter()
                    .cloned()
                    .collect::<ThinVec<_>>(),
            )))
        }
        (Value::Null, _, _) | (_, Value::Null, _) | (_, _, Value::Null) => Ok(Value::Null),
        _ => Err(String::from("Invalid array range parameters.")),
    }
}

#[inline]
pub(crate) const fn logical_xor(
    a: bool,
    b: bool,
) -> bool {
    (a && !b) || (!a && b)
}

pub fn evaluate_param(expr: &DynNode<ExprIR<Arc<String>>>) -> Result<Value, String> {
    match expr.data() {
        ExprIR::Constant(v) => Ok(v.clone()),
        ExprIR::List => Ok(Value::List(Arc::new(
            expr.children()
                .map(|c| evaluate_param(&c))
                .collect::<Result<ThinVec<_>, _>>()?,
        ))),
        ExprIR::Map => Ok(Value::Map(Arc::new(
            expr.children()
                .map(|ir| match ir.data() {
                    ExprIR::Constant(Value::String(key)) => {
                        Ok::<_, String>((key.clone(), evaluate_param(&ir.child(0))?))
                    }
                    _ => Err("Map parameter key must be a string".into()),
                })
                .collect::<Result<OrderMap<_, _>, _>>()?,
        ))),
        ExprIR::Negate => {
            let v = evaluate_param(&expr.child(0))?;
            match v {
                Value::Int(i) => Ok(Value::Int(i.checked_neg().ok_or_else(|| {
                    String::from("ArgumentError: integer overflow in unary minus")
                })?)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Ok(Value::Null),
            }
        }
        _ => Err(String::from("Invalid parameter expression.")),
    }
}
