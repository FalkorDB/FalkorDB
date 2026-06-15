//! Batch-mode procedure call operator — invokes built-in graph procedures.
//!
//! Implements Cypher `CALL db.procedure(args) YIELD outputs`.
//!
//! ```text
//!  Input rows (from child)
//!       │
//!  ┌────▼─────────────────────────┐
//!  │ for each row:                │
//!  │   eval argument expressions  │
//!  │   validate argument types    │
//!  │   invoke procedure function  │
//!  │   decode returned List rows  │
//!  │     to output Env rows       │
//!  └────┬─────────────────────────┘
//!       │
//!  output batches (up to BATCH_SIZE)
//! ```
//!
//! This operator streams output: each `next()` call pulls child rows, invokes
//! procedures, and emits up to one output batch without buffering the full
//! result set in memory. Only allowed in write queries when the procedure is
//! marked as write; returns an error for `GRAPH.RO_QUERY` on write procedures.

use std::sync::Arc;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    functions::{FnType, GraphFn},
    row::Row,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use thin_vec::ThinVec;

pub struct ProcedureCallOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    func: &'a Arc<GraphFn>,
    trees: &'a [QueryExpr<Variable>],
    output_var_ids: Vec<u32>,
    output_source_positions: Vec<usize>,
    pending_source_row: Option<usize>,
    pending_origin: u32,
    pending_proc_batch: Option<Batch<'static>>,
    pending_idx: usize,
    child_batch: Option<Batch<'a>>,
    child_active_pos: usize,
    child_exhausted: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ProcedureCallOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        func: &'a Arc<GraphFn>,
        trees: &'a [QueryExpr<Variable>],
        name_outputs: &'a [Variable],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<Self, String> {
        if !runtime.write && func.write {
            return Err(String::from(
                "graph.RO_QUERY is to be executed only on read-only queries",
            ));
        }
        let schema_outputs = match &func.fn_type {
            FnType::Procedure(cols) => cols.clone(),
            _ => Vec::new(),
        };

        let mut output_var_ids = Vec::with_capacity(name_outputs.len());
        let mut output_source_positions = Vec::with_capacity(name_outputs.len());
        for (yield_pos, output) in name_outputs.iter().enumerate() {
            output_var_ids.push(output.id);
            let source_pos = output
                .name
                .as_ref()
                .and_then(|name| schema_outputs.iter().position(|s| s == name.as_ref()))
                .unwrap_or(yield_pos);
            output_source_positions.push(source_pos);
        }

        Ok(Self {
            runtime,
            child,
            func,
            trees,
            output_var_ids,
            output_source_positions,
            pending_source_row: None,
            pending_origin: 0,
            pending_proc_batch: None,
            pending_idx: 0,
            child_batch: None,
            child_active_pos: 0,
            child_exhausted: false,
            idx,
        })
    }

    pub(crate) fn reset_state(&mut self) {
        self.pending_source_row = None;
        self.pending_origin = 0;
        self.pending_proc_batch = None;
        self.pending_idx = 0;
        self.child_batch = None;
        self.child_active_pos = 0;
        self.child_exhausted = false;
    }

    fn emit_pending_rows(
        &mut self,
        builder: &mut BatchBuilder,
    ) -> Result<(), String> {
        let (Some(source_row), Some(proc_batch), Some(batch)) = (
            self.pending_source_row,
            self.pending_proc_batch.as_ref(),
            self.child_batch.as_ref(),
        ) else {
            return Ok(());
        };

        while builder.len() < BATCH_SIZE && self.pending_idx < proc_batch.active_len() {
            let row_idx = proc_batch
                .selection()
                .map_or(self.pending_idx, |sel| sel[self.pending_idx] as usize);
            self.pending_idx += 1;
            builder.push_merged(batch, source_row, proc_batch, row_idx, self.pending_origin);
        }

        if self.pending_idx >= proc_batch.active_len() {
            self.pending_source_row = None;
            self.pending_origin = 0;
            self.pending_proc_batch = None;
            self.pending_idx = 0;
        }

        Ok(())
    }

    fn next_input_row(&mut self) -> Option<Result<usize, String>> {
        loop {
            if let Some(batch) = self.child_batch.as_ref()
                && self.child_active_pos < batch.active_len()
            {
                let row_idx = batch.selection().map_or(self.child_active_pos, |sel| {
                    sel[self.child_active_pos] as usize
                });
                self.child_active_pos += 1;
                return Some(Ok(row_idx));
            }

            self.child_batch = None;
            self.child_active_pos = 0;

            if self.child_exhausted {
                return None;
            }

            match self.child.next() {
                Some(Ok(batch)) => {
                    self.child_batch = Some(batch);
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.child_exhausted = true;
                    return None;
                }
            }
        }
    }
}

impl<'a> Iterator for ProcedureCallOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();

        loop {
            if let Err(e) = self.emit_pending_rows(&mut builder) {
                return Some(Err(e));
            }
            if builder.len() >= BATCH_SIZE {
                return Some(Ok(builder.finish()));
            }

            let source_row = match self.next_input_row() {
                Some(Ok(row)) => row,
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    if builder.is_empty() {
                        return None;
                    }
                    return Some(Ok(builder.finish()));
                }
            };

            let (args, origin) = {
                let Some(batch) = self.child_batch.as_ref() else {
                    return Some(Err("Procedure source batch missing".into()));
                };
                let input_env = BatchRow::new(batch, source_row);
                let args = match self
                    .trees
                    .iter()
                    .map(|ir| {
                        ExprEval::from_runtime(self.runtime).eval(
                            ir,
                            ir.root().idx(),
                            Some(&input_env),
                            None,
                        )
                    })
                    .collect::<Result<ThinVec<_>, _>>()
                {
                    Ok(args) => args,
                    Err(e) => return Some(Err(e)),
                };
                (args, batch.origin_row(source_row))
            };

            if let Err(e) = self.func.validate_args_type(&args) {
                return Some(Err(e));
            }

            let proc_batch = match self.func.call_procedure_batch(self.runtime, &args) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            let proc_batch = {
                let mut out_builder = BatchBuilder::new();
                for row_idx in proc_batch.active_indices() {
                    let mut row = Row::new();
                    for (var_id, source_pos) in self
                        .output_var_ids
                        .iter()
                        .zip(&self.output_source_positions)
                    {
                        let val = proc_batch
                            .value_at(*source_pos as u32, row_idx)
                            .unwrap_or(Value::Null);
                        row.insert_by_id(*var_id, val);
                    }
                    out_builder.push_row(&row);
                }
                out_builder.finish()
            };

            self.pending_source_row = Some(source_row);
            self.pending_origin = origin;
            self.pending_proc_batch = Some(proc_batch);
            self.pending_idx = 0;
        }
    }
}
