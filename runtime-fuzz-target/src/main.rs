// Dependency version duplicates are from transitive dependencies.
#![allow(clippy::multiple_crate_versions)]

use std::collections::HashMap;

use graph::{
    graph::{
        graph::Plan,
        graphblas::{GrB_Mode, GrB_init},
        mvcc_graph::MvccGraph,
    },
    runtime::{eval::evaluate_param, functions::init_functions, runtime::Runtime},
};

#[macro_use]
extern crate afl;

/// `WriteEscalation` for the fuzzer: there is no host and no locks, so becoming a
/// writer always succeeds. Returning `Err` instead would reject every mutating query
/// and starve the corpus of the write paths this target exists to exercise.
struct AlwaysWritable;

impl graph::locks::WriteEscalation for AlwaysWritable {
    fn upgrade_to_write(&self) -> Result<(), String> {
        Ok(())
    }
}

fn main() {
    unsafe {
        GrB_init(GrB_Mode::GrB_NONBLOCKING as _);
    }
    init_functions().expect("Failed to init functions");
    fuzz!(|data: &[u8]| {
        if let Ok(query) = std::str::from_utf8(data) {
            let g = MvccGraph::new(1024, 1024, 25, "fuzz_test");
            let Ok(Plan {
                plan, parameters, ..
            }) = g.read().borrow().get_plan(query)
            else {
                return;
            };
            let Ok(parameters) = parameters
                .into_iter()
                .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
                .collect::<Result<HashMap<_, _>, String>>()
            else {
                return;
            };
            let runtime = Runtime::new(
                g.read(),
                parameters,
                true,
                plan,
                false,
                String::new(),
                -1,
                false,
                None,
                0,
                None,
                &AlwaysWritable,
            );
            let _ = runtime.query();
        }
    });
}
