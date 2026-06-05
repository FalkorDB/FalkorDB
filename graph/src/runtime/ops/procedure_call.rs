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
//!  │   map returned List<Map>     │
//!  │     to output Env rows       │
//!  └────┬─────────────────────────┘
//!       │
//!  output batches (up to BATCH_SIZE)
//! ```
//!
//! This is a *blocking* operator: it consumes all input rows first, evaluates
//! argument expressions in each row's environment, invokes the procedure, and
//! maps the returned list of maps to output environments. Only allowed in
//! write queries when the procedure is marked as write; returns an error for
//! `GRAPH.RO_QUERY` on write procedures.

use std::sync::Arc;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    functions::GraphFn,
    row::{Row, RowView},
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
    name_outputs: &'a [Variable],
    pub(crate) batches: Option<std::vec::IntoIter<Batch<'a>>>,
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
        Ok(Self {
            runtime,
            child,
            func,
            trees,
            name_outputs,
            batches: None,
            idx,
        })
    }

    fn init_batches(&mut self) -> Result<(), String> {
        let mut all_envs: Vec<Row> = Vec::new();

        // Iterate over all input rows from child operator
        loop {
            match self.child.next() {
                Some(Ok(batch)) => {
                    for row_idx in batch.active_indices() {
                        let input_env = BatchRow::new(&batch, row_idx).to_owned_row();
                        // Evaluate arguments in the context of this input row
                        let args = self
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
                            .collect::<Result<ThinVec<_>, _>>()?;
                        self.func.validate_args_type(&args)?;
                        let res = self.func.func.call(self.runtime, &args)?;
                        match res {
                            Value::List(arr) => {
                                for v in arr.iter() {
                                    let mut env = input_env.clone();
                                    if let Value::Map(map) = v {
                                        for output in self.name_outputs {
                                            let field_name = output.name.as_ref().unwrap();
                                            let value =
                                                map.get(field_name).cloned().unwrap_or(Value::Null);
                                            env.insert(output, value);
                                        }
                                    }
                                    all_envs.push(env);
                                }
                            }
                            _ => return Err("Procedure must return a list".into()),
                        }
                    }
                }
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }

        let batches: Vec<Batch<'a>> = if all_envs.is_empty() {
            Vec::new()
        } else {
            let mut result = Vec::new();
            let mut builder = BatchBuilder::new();
            for env in all_envs {
                builder.push_row(&env);
                if builder.len() >= BATCH_SIZE {
                    result.push(std::mem::take(&mut builder).finish());
                }
            }
            if !builder.is_empty() {
                result.push(builder.finish());
            }
            result
        };
        self.batches = Some(batches.into_iter());
        Ok(())
    }
}

impl<'a> Iterator for ProcedureCallOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Initialize batches on first call.
        if self.batches.is_none()
            && let Err(e) = self.init_batches()
        {
            // Mark initialization as done so we don't retry on subsequent calls.
            self.batches = Some(Vec::new().into_iter());
            return Some(Err(e));
        }

        self.batches.as_mut().unwrap().next().map(Ok)
    }
}
