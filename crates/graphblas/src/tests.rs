//! Behavioural tests of the typed layer, run against the real library.
//!
//! `GrB_init` is process-wide, so every test goes through [`setup`], and the
//! tests that need their own process (use before `init`) live
//! under `tests/`.

#![allow(
    clippy::many_single_char_names,
    reason = "`a`, `b`, `c`, `u`, `v`, `w` are what the GraphBLAS spec calls the operands"
)]

use std::{
    ffi::{CStr, c_char},
    os::raw::c_int,
    sync::Once,
};

use crate::{
    JitControl, Lazy,
    init::{init, is_initialized, set_jit},
    ops::*,
    sys::{self, GrB_Index, GrB_Info},
    types::{GrbType, Pattern},
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

unsafe extern "C" fn edge_weight(
    z: *mut f64,
    x: *const Edge,
) {
    // SAFETY: GraphBLAS passes valid pointers of the declared types.
    unsafe { *z = (*x).weight };
}

unsafe extern "C" fn heavier(
    z: *mut Edge,
    x: *const Edge,
    y: *const Edge,
) {
    // SAFETY: as above.
    unsafe { *z = if (*x).weight >= (*y).weight { *x } else { *y } };
}

unsafe extern "C" fn id_above(
    z: *mut bool,
    x: *const Edge,
    _i: GrB_Index,
    _j: GrB_Index,
    y: *const u64,
) {
    // SAFETY: as above.
    unsafe { *z = (*x).id > *y };
}

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

crate::user_ops! {
    /// `z = x.weight`.
    // SAFETY: `edge_weight` reads `x` and then writes a whole `f64`.
    static EDGE_WEIGHT: UnaryOp<Edge, f64> = unsafe { UnaryOp::new(edge_weight) };
    // SAFETY: `heavier` picks one input and copies it to `z`, which may be
    // either of them.
    static HEAVIER: BinaryOp<Edge, Edge, Edge> = unsafe { BinaryOp::new(heavier) };
    static EDGE_MAX: Monoid<Edge> = Monoid::new(
        &HEAVIER,
        Edge {
            weight: f64::NEG_INFINITY,
            id: 0,
        },
    );
    static HEAVIEST_PATH: Semiring<Edge, Edge, Edge> = Semiring::new(&EDGE_MAX, &HEAVIER);
    // SAFETY: `id_above` reads `x` and `y` and then writes a whole `bool`.
    static ID_ABOVE: IndexUnaryOp<Edge, u64, bool> = unsafe { IndexUnaryOp::new(id_above) };
}

#[test]
fn user_defined_operators_are_typed_and_static() {
    setup();
    let weight = &EDGE_WEIGHT;
    let heavier_op = &HEAVIER;
    let edge_max = &EDGE_MAX;
    let heaviest_path = &HEAVIEST_PATH;
    assert!(!ID_ABOVE.as_raw().is_null());
    assert_eq!(edge_max.op().as_raw(), heavier_op.as_raw());
    assert_eq!(edge_max.identity().id, 0);
    assert_eq!(heaviest_path.add().as_raw(), edge_max.as_raw());
    assert_eq!(heaviest_path.multiply().as_raw(), heavier_op.as_raw());
    assert_eq!(
        EDGE_WEIGHT.as_raw(),
        weight.as_raw(),
        "a static is created once"
    );
}

// ───────────────────────────── built-ins ─────────────────────────────

#[test]
fn builtins_are_read_once_and_non_null() {
    setup();
    let first = ANY_PAIR_BOOL.as_raw();
    assert!(!first.is_null());
    assert_eq!(Lazy::force(&ANY_PAIR_BOOL).as_raw(), first);
    // SAFETY: a by-value read of a handle `GrB_init` set.
    assert_eq!(first, unsafe { sys::GxB_ANY_PAIR_BOOL });
}

/// The GraphBLAS name of an operator's input or output type, through
/// `GrB_get` (`GrB_INP0_TYPE_STRING`, ...); `None` when the object has no
/// type in that slot (positional operators never read `x`).
fn reported_name<H>(
    get: unsafe extern "C" fn(H, *mut c_char, c_int) -> GrB_Info,
    handle: H,
    field: sys::GxB_Option_Field,
) -> Option<String> {
    let mut buf = [0 as c_char; sys::GxB_MAX_NAME_LEN as usize];
    // SAFETY: a live handle and a buffer of the size GraphBLAS requires.
    let info = unsafe { get(handle, buf.as_mut_ptr(), field as c_int) };
    if info != GrB_Info::GrB_SUCCESS {
        return None;
    }
    // SAFETY: GraphBLAS wrote a NUL-terminated string into `buf`.
    let name = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    (!name.is_empty()).then_some(name)
}

const X: sys::GxB_Option_Field = sys::GxB_Option_Field::GrB_INP0_TYPE_STRING;
const Y: sys::GxB_Option_Field = sys::GxB_Option_Field::GrB_INP1_TYPE_STRING;
const Z: sys::GxB_Option_Field = sys::GxB_Option_Field::GrB_OUTP_TYPE_STRING;

/// Every built-in's declared domains against what GraphBLAS reports for the
/// object. A concrete declaration must match the reported type; a slot
/// GraphBLAS reports no type for (positional operators) must be declared
/// `Pattern`. A `Pattern` declaration on a slot with a reported type is not
/// checked: `PAIR`, `ONE` and `ONEB` carry a type and still never read it.
#[test]
fn builtin_declarations_match_graphblas() {
    setup();
    let mut problems = Vec::new();
    for (symbol, declared) in all_declared() {
        for (slot, expected, reported) in declared.check() {
            match (expected, reported) {
                (Some(e), Some(r)) if e != r => {
                    problems.push(format!("{symbol}: {slot} declared {e}, is {r}"));
                }
                (Some(e), None) => {
                    problems.push(format!("{symbol}: {slot} declared {e}, has no type"));
                }
                _ => {}
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Reflective view of a built-in, for the declaration check.
pub trait Declared {
    /// `(slot, declared type name or None for Pattern, reported type name or
    /// None for a slot without one)`.
    fn check(&self) -> Vec<Check>;
}

type Check = (&'static str, Option<&'static str>, Option<String>);

impl<X: Domain, Z: Domain> Declared for UnaryOp<X, Z> {
    fn check(&self) -> Vec<Check> {
        vec![
            (
                "x",
                X::DECLARED,
                reported_name(sys::GrB_UnaryOp_get_String, self.as_raw(), X),
            ),
            (
                "z",
                Z::DECLARED,
                reported_name(sys::GrB_UnaryOp_get_String, self.as_raw(), Z),
            ),
        ]
    }
}

impl<X: Domain, Y: Domain, Z: Domain> Declared for BinaryOp<X, Y, Z> {
    fn check(&self) -> Vec<Check> {
        vec![
            (
                "x",
                X::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, self.as_raw(), X),
            ),
            (
                "y",
                Y::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, self.as_raw(), Y),
            ),
            (
                "z",
                Z::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, self.as_raw(), Z),
            ),
        ]
    }
}

impl<X: Domain, Y: Domain, Z: Domain> Declared for IndexUnaryOp<X, Y, Z> {
    fn check(&self) -> Vec<Check> {
        vec![
            (
                "x",
                X::DECLARED,
                reported_name(sys::GrB_IndexUnaryOp_get_String, self.as_raw(), X),
            ),
            (
                "y",
                Y::DECLARED,
                reported_name(sys::GrB_IndexUnaryOp_get_String, self.as_raw(), Y),
            ),
            (
                "z",
                Z::DECLARED,
                reported_name(sys::GrB_IndexUnaryOp_get_String, self.as_raw(), Z),
            ),
        ]
    }
}

impl<T: Domain> Declared for Monoid<T> {
    fn check(&self) -> Vec<Check> {
        let op = self.op();
        vec![
            (
                "x",
                T::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, op.as_raw(), X),
            ),
            (
                "y",
                T::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, op.as_raw(), Y),
            ),
            (
                "z",
                T::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, op.as_raw(), Z),
            ),
        ]
    }
}

impl<X: Domain, Y: Domain, Z: Domain> Declared for Semiring<X, Y, Z> {
    fn check(&self) -> Vec<Check> {
        let add = self.add().op();
        let mul = self.multiply();
        vec![
            (
                "add z",
                Z::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, add.as_raw(), Z),
            ),
            (
                "mult x",
                X::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, mul.as_raw(), X),
            ),
            (
                "mult y",
                Y::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, mul.as_raw(), Y),
            ),
            (
                "mult z",
                Z::DECLARED,
                reported_name(sys::GrB_BinaryOp_get_String, mul.as_raw(), Z),
            ),
        ]
    }
}

/// What a type parameter of a built-in declaration claims: a GraphBLAS type
/// name, or nothing for [`Pattern`].
pub trait Domain {
    const DECLARED: Option<&'static str>;
}

impl Domain for Pattern {
    const DECLARED: Option<&'static str> = None;
}

impl Domain for crate::types::BuiltinPattern {
    const DECLARED: Option<&'static str> = None;
}

impl<T: crate::BuiltinType> Domain for T {
    const DECLARED: Option<&'static str> = Some(T::NAME);
}

#[test]
fn raw_pointers_of_user_types_are_distinct_from_builtins() {
    setup();
    // SAFETY: by-value read of a handle `GrB_init` set.
    let bool_type = unsafe { sys::GrB_BOOL };
    assert_eq!(bool::grb_type(), bool_type);
    assert_ne!(Edge::grb_type(), bool_type);
    assert_eq!(Edge::grb_type(), Edge::grb_type(), "created once");
}

#[test]
fn a_semiring_can_multiply_with_an_operator_that_reads_no_values() {
    setup();
    let semiring: Semiring<Pattern, Pattern, bool> =
        Semiring::new(&LOR_MONOID_BOOL, &PAIR_BOOL).unwrap();
    assert!(!semiring.add().identity());
}
