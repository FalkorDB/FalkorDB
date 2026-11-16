#![no_main]

use graph::{
    fuzz_support::init_index_support,
    graph::{
        graph::{Graph, Plan},
        graphblas::{GrB_Mode, GrB_init},
        mvcc_graph::MvccGraph,
    },
    runtime::{functions::init_functions, runtime::Runtime},
};
use libfuzzer_sys::{Corpus, fuzz_target};

/// `WriteEscalation` for the fuzzer: there is no host and no locks, so becoming a
/// writer always succeeds. Returning `Err` instead would reject every mutating
/// query and starve the corpus of the write paths this target exists to exercise.
struct AlwaysWritable;

impl graph::locks::WriteEscalation for AlwaysWritable {
    fn upgrade_to_write(
        &self,
        _fork: &mut Graph,
    ) -> Result<(), String> {
        Ok(())
    }
}

fuzz_target!(init: {
        unsafe {
            GrB_init(GrB_Mode::GrB_NONBLOCKING as _);
        }
        init_index_support();
        init_functions().expect("Failed to init functions");
    },|data: &[u8]| -> Corpus {
        let g = MvccGraph::new(1024, 1024, 25, "fuzz_test");
    std::str::from_utf8(data).map_or(Corpus::Reject, |query| {
        let Ok(Plan {
            plan, parameters, ..
        }) = g.read().get_plan(query)
        else {
            return Corpus::Reject;
        };
        // 100ms query timeout: the fuzzer otherwise finds semantically huge but
        // valid queries (e.g. UNWIND range(..1M) CREATE 15 nodes per row) that
        // blow past libFuzzer's 2048MB rss limit. Timed-out inputs return Err
        // and are rejected from the corpus.
        let runtime = Runtime::new(
            g.fork(),
            parameters,
            plan,
            false,
            String::new(),
            -1,
            false,
            Some(100),
            0,
            None,
            &AlwaysWritable,
        );
        match runtime.query() {
            Ok(_) => Corpus::Keep,
            _ => Corpus::Reject,
        }
    })
});
