//! Batch-mode merge operator — implements Cypher `MERGE` (match-or-create).
//!
//! For each input batch, runs the match sub-plan once with all active rows as a
//! multi-row argument batch. Uses `origin_row` on output envs to group matches
//! by input row.
//!
//! ```text
//!  Input row ──► run match sub-plan
//!                      │
//!              ┌───────┴───────┐
//!              │               │
//!          has matches?    no matches?
//!              │               │
//!        ON MATCH SET     CREATE pattern
//!              │           + ON CREATE SET
//!              ▼               ▼
//!         output row       output row
//! ```
//!
//! For input rows with matches, applies `ON MATCH SET`; for rows without
//! matches, creates the pattern and applies `ON CREATE SET`. A pattern
//! hash cache prevents duplicate creates within the same query — if the
//! same pattern (same labels, attributes, endpoints) was already created
//! by an earlier row, the cached entity IDs are reused and `ON MATCH SET`
//! is applied instead.

use std::cell::OnceCell;
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::graph::graph::LabelId;
use crate::parser::ast::{QueryGraph, SetItem, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Pending merge results for a single input row (multiple matches to drain lazily).
struct PendingMerge<'a> {
    /// The input env to merge with each match result.
    env: Env<'a>,
    /// Remaining match envs to process.
    matches: VecDeque<Env<'a>>,
}

pub struct MergeOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<PendingMerge<'a>>,
    merge_child_idx: NodeIdx<Dyn<IR>>,
    pattern: &'a QueryGraph<Arc<String>, Arc<String>, Variable>,
    resolved_pattern: OnceCell<QueryGraph<Arc<String>, LabelId, Variable>>,
    on_create_set_items: &'a [SetItem<Arc<String>, Variable>],
    resolved_on_create_set_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    on_match_set_items: &'a [SetItem<Arc<String>, Variable>],
    resolved_on_match_set_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    is_error: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> MergeOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        pattern: &'a QueryGraph<Arc<String>, Arc<String>, Variable>,
        on_create_set_items: &'a [SetItem<Arc<String>, Variable>],
        on_match_set_items: &'a [SetItem<Arc<String>, Variable>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let merge_child_idx = if runtime.plan.node(idx).num_children() == 1 {
            runtime.plan.node(idx).child(0).idx()
        } else {
            runtime.plan.node(idx).child(1).idx()
        };

        Self {
            runtime,
            child,
            pending: VecDeque::new(),
            merge_child_idx,
            pattern,
            resolved_pattern: OnceCell::new(),
            on_create_set_items,
            resolved_on_create_set_items: OnceCell::new(),
            on_match_set_items,
            resolved_on_match_set_items: OnceCell::new(),
            is_error: false,
            idx,
        }
    }

    fn resolve_pattern(&self) -> &QueryGraph<Arc<String>, LabelId, Variable> {
        self.resolved_pattern
            .get_or_init(|| self.runtime.resolve_pattern(self.pattern))
    }

    fn resolve_on_create_set_items(&self) -> &Vec<SetItem<LabelId, Variable>> {
        self.resolved_on_create_set_items
            .get_or_init(|| self.runtime.resolve_set_items(self.on_create_set_items))
    }

    fn resolve_on_match_set_items(&self) -> &Vec<SetItem<LabelId, Variable>> {
        self.resolved_on_match_set_items
            .get_or_init(|| self.runtime.resolve_set_items(self.on_match_set_items))
    }

    fn do_create_fallback(
        &self,
        vars: Env<'a>,
    ) -> Result<Env<'a>, String> {
        let resolved_pattern = self.resolve_pattern();
        let pattern_hash = self.compute_merge_pattern_hash(resolved_pattern, &vars)?;

        let merge_cache = self.runtime.merge_pattern_cache.borrow_mut();

        if let Some(cached_vars) = merge_cache.get(&pattern_hash) {
            // Pattern already created, apply ON MATCH and return cached vars
            let mut vars = vars;
            for (id, value) in cached_vars {
                vars.insert_by_id(*id, value.clone());
            }
            drop(merge_cache);

            let resolved = self.resolve_on_match_set_items();
            let batch = Batch::from_envs(vec![vars]);
            self.runtime.set_batch(resolved, &batch)?;
            let mut envs_vec = batch.into_envs();
            Ok(envs_vec.pop().unwrap())
        } else {
            // Pattern not yet created, create it
            drop(merge_cache);

            let mut batch = Batch::from_envs(vec![vars]);
            self.runtime.create_batch(resolved_pattern, &mut batch)?;

            // Cache only the created entity bindings (node/relationship IDs)
            let env_ref = batch.env_ref(0);
            let pattern_vars: Vec<(u32, Value)> = resolved_pattern
                .nodes()
                .iter()
                .map(|n| {
                    (
                        n.alias.id,
                        env_ref.get(&n.alias).cloned().unwrap_or(Value::Null),
                    )
                })
                .chain(resolved_pattern.relationships().iter().map(|r| {
                    (
                        r.alias.id,
                        env_ref.get(&r.alias).cloned().unwrap_or(Value::Null),
                    )
                }))
                .collect();
            self.runtime
                .merge_pattern_cache
                .borrow_mut()
                .insert(pattern_hash, pattern_vars);

            let resolved = self.resolve_on_create_set_items();
            self.runtime.set_batch(resolved, &batch)?;
            let mut envs_vec = batch.into_envs();
            Ok(envs_vec.pop().unwrap())
        }
    }

    fn compute_merge_pattern_hash(
        &self,
        pattern: &QueryGraph<Arc<String>, LabelId, Variable>,
        vars: &Env<'a>,
    ) -> Result<u64, String> {
        let mut hasher = DefaultHasher::new();

        // Hash nodes in the pattern
        for node in pattern.nodes() {
            if let Some(value) = vars.get(&node.alias) {
                value.hash(&mut hasher);
            } else {
                for label in node.labels.iter() {
                    label.hash(&mut hasher);
                }
                let attrs = ExprEval::from_runtime(self.runtime).eval(
                    &node.attrs,
                    node.attrs.root().idx(),
                    Some(vars),
                    None,
                )?;

                if let Value::Map(ref map) = attrs {
                    for (key, value) in map.iter() {
                        if matches!(value, Value::Null) {
                            return Err(format!(
                                "Cannot merge node using null property value for key '{key}'"
                            ));
                        }
                    }
                }

                attrs.hash(&mut hasher);
            }
        }

        // Hash relationships in the pattern
        for rel in pattern.relationships() {
            rel.types.hash(&mut hasher);

            if let Some(value) = vars.get(&rel.from.alias) {
                value.hash(&mut hasher);
            }
            if let Some(value) = vars.get(&rel.to.alias) {
                value.hash(&mut hasher);
            }

            let attrs = ExprEval::from_runtime(self.runtime).eval(
                &rel.attrs,
                rel.attrs.root().idx(),
                Some(vars),
                None,
            )?;

            if let Value::Map(ref map) = attrs {
                for (key, value) in map.iter() {
                    if matches!(value, Value::Null) {
                        return Err(format!(
                            "Cannot merge relationship using null property value for key '{key}'"
                        ));
                    }
                }
            }

            attrs.hash(&mut hasher);
        }

        Ok(hasher.finish())
    }

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending are exhausted.
    fn drain_pending(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        while envs.len() < BATCH_SIZE && !self.pending.is_empty() {
            let p = self.pending.front_mut().unwrap();

            if let Some(match_env) = p.matches.pop_front() {
                let mut vars = p.env.clone_pooled(self.runtime.env_pool);
                vars.merge(&match_env);
                let resolved = self.resolve_on_match_set_items();
                let result_batch = Batch::from_envs(vec![vars]);
                self.runtime.set_batch(resolved, &result_batch)?;
                let mut envs_vec = result_batch.into_envs();
                envs.push(envs_vec.pop().unwrap());
            } else {
                self.pending.pop_front();
            }
        }
        Ok(())
    }
}

impl<'a> Iterator for MergeOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_error {
            return None;
        }

        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover match results from previous call.
        if let Err(e) = self.drain_pending(&mut envs) {
            self.is_error = true;
            return Some(Err(e));
        }

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    self.is_error = true;
                    return Some(Err(e));
                }
                None => break,
            };

            // Build argument batch with origin_row stamped.
            let input_envs: Vec<Env<'a>> = batch
                .active_env_iter()
                .enumerate()
                .map(|(i, env)| {
                    let mut e = env.clone_pooled(self.runtime.env_pool);
                    e.origin_row = i as u32;
                    e
                })
                .collect();

            let arg_envs: Vec<Env<'a>> = input_envs
                .iter()
                .map(|e| e.clone_pooled(self.runtime.env_pool))
                .collect();

            // Create ONE match subtree for all input rows.
            let mut subtree = match self.runtime.run_batch(self.merge_child_idx) {
                Ok(s) => s,
                Err(e) => {
                    self.is_error = true;
                    return Some(Err(e));
                }
            };
            subtree.set_argument_batch(Batch::from_envs(arg_envs));

            // Materialize all matches grouped by origin_row.
            let num_inputs = input_envs.len();
            let mut match_groups: Vec<Vec<Env<'a>>> = (0..num_inputs).map(|_| Vec::new()).collect();

            for sub_result in subtree.by_ref() {
                match sub_result {
                    Ok(sub_batch) => {
                        for env in sub_batch.active_env_iter() {
                            let origin = env.origin_row as usize;
                            match_groups[origin].push(env.clone_pooled(self.runtime.env_pool));
                        }
                    }
                    Err(e) => {
                        self.is_error = true;
                        return Some(Err(e));
                    }
                }
            }
            drop(subtree);

            // Process each input row in order.
            for (i, input_env) in input_envs.iter().enumerate() {
                let matches = std::mem::take(&mut match_groups[i]);

                if matches.is_empty() {
                    // No matches found, do create fallback.
                    match self.do_create_fallback(input_env.clone_pooled(self.runtime.env_pool)) {
                        Ok(result_env) => envs.push(result_env),
                        Err(e) => {
                            self.is_error = true;
                            return Some(Err(e));
                        }
                    }
                } else {
                    // Check if all pattern variables are already bound.
                    // All nodes must be bound, and every *named* relationship
                    // alias must also be bound. Anonymous relationships
                    // (_anon_* prefix) are not individually tracked, so when
                    // all nodes are bound the pattern is fully constrained.
                    // Only when a user-named relationship variable is unbound
                    // do we need to iterate all matches.
                    let pattern = self.resolve_pattern();
                    let all_vars_bound = pattern
                        .nodes()
                        .iter()
                        .all(|node| input_env.get(&node.alias).is_some())
                        && pattern
                            .relationships()
                            .iter()
                            .filter(|rel| {
                                rel.alias
                                    .name
                                    .as_ref()
                                    .is_some_and(|n| !n.starts_with("_anon_"))
                            })
                            .all(|rel| input_env.get(&rel.alias).is_some());

                    if all_vars_bound {
                        // Only first match needed.
                        let first = &matches[0];
                        let mut vars = input_env.clone_pooled(self.runtime.env_pool);
                        vars.merge(first);
                        let resolved = self.resolve_on_match_set_items();
                        let result_batch = Batch::from_envs(vec![vars]);
                        match self.runtime.set_batch(resolved, &result_batch) {
                            Ok(()) => {
                                let mut envs_vec = result_batch.into_envs();
                                envs.push(envs_vec.pop().unwrap());
                            }
                            Err(e) => {
                                self.is_error = true;
                                return Some(Err(e));
                            }
                        }
                    } else {
                        // Process first match inline, queue remaining for lazy drain.
                        let mut match_iter = matches.into_iter();
                        let first = match_iter.next().unwrap();

                        let mut vars = input_env.clone_pooled(self.runtime.env_pool);
                        vars.merge(&first);
                        let resolved = self.resolve_on_match_set_items();
                        let result_batch = Batch::from_envs(vec![vars]);
                        match self.runtime.set_batch(resolved, &result_batch) {
                            Ok(()) => {
                                let mut envs_vec = result_batch.into_envs();
                                envs.push(envs_vec.pop().unwrap());
                            }
                            Err(e) => {
                                self.is_error = true;
                                return Some(Err(e));
                            }
                        }

                        let remaining: VecDeque<Env<'a>> = match_iter.collect();
                        if !remaining.is_empty() {
                            self.pending.push_back(PendingMerge {
                                env: input_env.clone_pooled(self.runtime.env_pool),
                                matches: remaining,
                            });
                        }
                    }
                }
            }

            if let Err(e) = self.drain_pending(&mut envs) {
                self.is_error = true;
                return Some(Err(e));
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
