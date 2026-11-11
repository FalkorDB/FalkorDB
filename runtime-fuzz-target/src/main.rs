// Dependency version duplicates are from transitive dependencies.
#![allow(clippy::multiple_crate_versions)]

use libc::{calloc, free, malloc, realloc};

use graph::{
    graph::{
        graph::{Graph, Plan},
        graphblas::matrix,
        mvcc_graph::MvccGraph,
    },
    runtime::{functions::init_functions, runtime::Runtime},
};

#[macro_use]
extern crate afl;

/// `WriteEscalation` for the fuzzer: there is no host and no locks, so becoming a
/// writer always succeeds. Returning `Err` instead would reject every mutating query
/// and starve the corpus of the write paths this target exists to exercise.
struct AlwaysWritable;

impl graph::locks::WriteEscalation for AlwaysWritable {
    fn upgrade_to_write(
        &self,
        _fork: &mut Graph,
    ) -> Result<(), String> {
        Ok(())
    }
}

fn main() {
    // SAFETY: GraphBLAS allocates with libc, and this binary's global
    // allocator is `System`, which frees with libc `free` on unix.
    unsafe {
        matrix::init(Some(malloc), Some(calloc), Some(realloc), Some(free))
            .expect("GraphBLAS initializes");
    }
    init_functions().expect("Failed to init functions");
    fuzz!(|data: &[u8]| {
        if let Ok(query) = std::str::from_utf8(data) {
            let g = MvccGraph::new(1024, 1024, 25, "fuzz_test");
            let Ok(Plan {
                plan, parameters, ..
            }) = g.read().get_plan(query)
            else {
                return;
            };
            let runtime = Runtime::new(
                g.fork(),
                parameters,
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
