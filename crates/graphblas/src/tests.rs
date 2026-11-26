//! Behavioural tests of the typed layer, run against the real library.
//!
//! `GrB_init` is process-wide, so every test goes through [`setup`], and the
//! tests that need their own process (use before `init`) live
//! under `tests/`.

use std::sync::Once;

use crate::{
    JitControl,
    init::{init, is_initialized, set_jit},
    types::GrbType,
};
use std::sync::mpsc;
use std::thread;

fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // OpenMP binds its runtime to the thread that first uses it. With
        // the statically linked libomp of the Linux builds, a later
        // `omp critical` (every `GrB_Global_get` and `set` has one) crashes
        // once that thread has exited. Tests run on short-lived threads, so
        // GraphBLAS is started on one that never ends.
        let (started, wait) = mpsc::channel();
        thread::spawn(move || {
            init(None).expect("GrB_init");
            set_jit(JitControl::Off).expect("JIT off");
            started.send(()).expect("the test is waiting");
            loop {
                thread::park();
            }
        });
        wait.recv().expect("GraphBLAS started");
    });
    assert!(is_initialized());
}

// ───────────────────────────── user-defined objects ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Edge {
    weight: f64,
    id: u64,
}
crate::user_type!(Edge);

#[test]
fn user_defined_type_round_trips_by_value() {
    setup();
    assert_eq!(Edge::NAME, "Edge");
    let e = Edge {
        weight: 1.5,
        id: 42,
    };
    assert_eq!((e.weight, e.id), (1.5, 42));
    assert_eq!(
        size_of::<Edge>(),
        16,
        "#[repr(C)]: no padding between the fields"
    );
}

#[test]
fn a_user_type_is_created_once() {
    setup();
    assert!(!Edge::grb_type().is_null());
    assert_eq!(Edge::grb_type(), Edge::grb_type(), "created once");
}
