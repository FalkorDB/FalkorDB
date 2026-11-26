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
    AsBool, Desc, Error, ErrorKind, Indices, JitControl, Lazy, Mask, Matrix, Output, Scalar,
    Sparsity,
    init::{init, is_initialized, set_jit},
    mask::{MaskMode, descriptor},
    matrix, mxm,
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

fn matrix<T: GrbType>(
    nrows: GrB_Index,
    ncols: GrB_Index,
    entries: &[(GrB_Index, GrB_Index, T)],
) -> Matrix<T> {
    let mut m = Matrix::new(nrows, ncols).unwrap();
    for &(i, j, x) in entries {
        m.set_element(i, j, x).unwrap();
    }
    m
}

/// Every entry, sorted by position.
fn entries<T: GrbType + PartialOrd>(m: &Matrix<T>) -> Vec<(GrB_Index, GrB_Index, T)> {
    let (rows, cols, vals) = m.extract_tuples().unwrap();
    let mut out: Vec<_> = rows
        .into_iter()
        .zip(cols)
        .zip(vals)
        .map(|((i, j), x)| (i, j, x))
        .collect();
    out.sort_by_key(|e| (e.0, e.1));
    out
}

// ───────────────────────────── handles ─────────────────────────────

#[test]
fn the_linked_library_is_the_release_the_bindings_describe() {
    setup();
    let version = |field: sys::GxB_Option_Field| {
        let mut part = -1;
        // SAFETY: the global object, a valid out-pointer and one of its
        // integer fields.
        let info =
            unsafe { sys::GrB_Global_get_INT32(sys::GrB_GLOBAL, &raw mut part, field as c_int) };
        assert_eq!(info, GrB_Info::GrB_SUCCESS);
        u32::try_from(part).expect("a version part is not negative")
    };
    assert_eq!(
        [
            version(sys::GxB_Option_Field::GrB_LIBRARY_VER_MAJOR),
            version(sys::GxB_Option_Field::GrB_LIBRARY_VER_MINOR),
            version(sys::GxB_Option_Field::GrB_LIBRARY_VER_PATCH),
        ],
        [
            sys::GxB_IMPLEMENTATION_MAJOR,
            sys::GxB_IMPLEMENTATION_MINOR,
            sys::GxB_IMPLEMENTATION_SUB,
        ],
        "regenerate crates/graphblas-sys/src/bindings.rs for this GraphBLAS"
    );
}

#[test]
fn new_matrix_is_empty_with_its_dimensions() {
    setup();
    let m = Matrix::<u64>::new(3, 4).unwrap();
    assert_eq!((m.nrows(), m.ncols(), m.nvals()), (3, 4, 0));
    assert_eq!(
        format!("{m:?}").split(" @ ").next(),
        Some("Matrix<GrB_UINT64>(3x4")
    );
}

#[test]
fn set_extract_remove_contains() {
    setup();
    let mut m = Matrix::<f64>::new(2, 2).unwrap();
    m.set_element(1, 0, 2.5).unwrap();
    assert_eq!(m.extract_element(1, 0).unwrap(), Some(2.5));
    assert_eq!(m.extract_element(0, 0).unwrap(), None);
    assert!(m.contains(1, 0).unwrap());
    assert!(!m.contains(0, 1).unwrap());
    m.remove_element(1, 0).unwrap();
    assert_eq!(m.nvals(), 0);
    assert!(matches!(
        m.set_element(2, 0, 1.0),
        Err(Error {
            kind: ErrorKind::InvalidIndex,
            ..
        })
    ));
}

#[test]
fn build_and_extract_tuples_round_trip() {
    setup();
    let mut m = Matrix::<i64>::new(4, 4).unwrap();
    m.build(&[3, 0, 2], &[1, 0, 2], &[30, 1, 20], None).unwrap();
    assert_eq!(entries(&m), vec![(0, 0, 1), (2, 2, 20), (3, 1, 30)]);
}

#[test]
fn build_duplicates_need_a_dup_operator() {
    setup();
    let mut m = Matrix::<u64>::new(2, 2).unwrap();
    let err = m.build(&[0, 0], &[1, 1], &[1, 2], None).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidValue);

    let mut m = Matrix::<u64>::new(2, 2).unwrap();
    m.build(&[0, 0], &[1, 1], &[1, 2], Some(&PLUS_UINT64))
        .unwrap();
    assert_eq!(entries(&m), vec![(0, 1, 3)]);
}

#[test]
fn build_iso_shares_one_value() {
    setup();
    let mut m = Matrix::<bool>::new(3, 3).unwrap();
    m.build_iso(&[0, 1, 1], &[0, 1, 1], true).unwrap();
    assert!(m.is_iso());
    assert_eq!(entries(&m), vec![(0, 0, true), (1, 1, true)]);
}

#[test]
fn dup_clear_resize() {
    setup();
    let a = matrix(3, 3, &[(0, 0, 1u64), (2, 2, 2)]);
    let mut b = a.dup().unwrap();
    assert_eq!(entries(&b), entries(&a));
    b.resize(1, 1).unwrap();
    assert_eq!(entries(&b), vec![(0, 0, 1)]);
    b.clear().unwrap();
    assert_eq!(b.nvals(), 0);
    assert_eq!(a.nvals(), 2, "the copy is independent");
}

#[test]
fn wait_finishes_pending_work() {
    setup();
    let mut m = Matrix::<u64>::new(10, 10).unwrap();
    for i in 0..10 {
        m.set_element(i, i, i).unwrap();
    }
    m.wait(crate::WaitMode::Materialize).unwrap();
    assert!(!m.is_pending());
    assert_eq!(m.nvals(), 10);
    assert!(m.memory_usage() > 0);
}

#[test]
fn storage_settings_round_trip() {
    setup();
    let mut m = Matrix::<f64>::new(4, 4).unwrap();
    assert_eq!(m.orientation(), crate::Orientation::ByRow);
    m.set_orientation(crate::Orientation::ByCol).unwrap();
    assert_eq!(m.orientation(), crate::Orientation::ByCol);

    matrix::assign_scalar(
        &mut m,
        Mask::NONE,
        None,
        1.0,
        Indices::All,
        Indices::All,
        Desc::NONE,
    )
    .unwrap();
    m.set_sparsity(Sparsity::Full).unwrap();
    m.wait(crate::WaitMode::Materialize).unwrap();
    assert_eq!(m.sparsity(), Sparsity::Full);
    m.set_sparsity(Sparsity::Auto).unwrap();

    m.set_index_width_hint(
        crate::IntegerWidth::Bits32,
        crate::IntegerWidth::Bits32,
        crate::IntegerWidth::Bits32,
    )
    .unwrap();
    let _ = (m.row_index_width(), m.col_index_width(), m.offset_width());
}

#[test]
fn into_raw_and_from_raw_hand_the_handle_over() {
    setup();
    let m = matrix(2, 2, &[(1, 1, 7u64)]);
    let raw = m.into_raw();
    // SAFETY: `raw` came from a `Matrix<u64>` that gave up ownership.
    let m = unsafe { Matrix::<u64>::from_raw(raw) };
    assert_eq!(m.extract_element(1, 1).unwrap(), Some(7));
}

#[test]
fn scalar_round_trip() {
    setup();
    let mut s = Scalar::<u64>::new().unwrap();
    assert_eq!(s.get().unwrap(), None);
    s.set(9).unwrap();
    assert_eq!(s.get().unwrap(), Some(9));
    s.clear().unwrap();
    assert_eq!(s.get().unwrap(), None);
    assert_eq!(Scalar::from_value(3i64).unwrap().get().unwrap(), Some(3));
}

// ───────────────────────────── masks and descriptors ─────────────────────────────

#[test]
fn as_bool_lets_a_counted_matrix_mask_by_value() {
    setup();
    // counts(0,0) = 0 and counts(1,1) = 2: a structural mask keeps both
    // positions, a valued one through `AsBool` keeps only the non-zero.
    let counts = matrix::<u64>(2, 2, &[(0, 0, 0), (1, 1, 2)]);
    let a = matrix::<u64>(2, 2, &[(0, 0, 7), (1, 1, 8), (0, 1, 9)]);

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::structural(&counts), None, &a, Desc::T0).unwrap();
    assert_eq!(entries(&c), vec![(0, 0, 7), (1, 1, 8)]);

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::valued(AsBool(&counts)), None, &a, Desc::T0).unwrap();
    assert_eq!(entries(&c), vec![(1, 1, 8)], "the stored zero is out");

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(
        &mut c,
        Mask::valued_complement(AsBool(&counts)),
        None,
        &a,
        Desc::T0,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 0, 7), (0, 1, 9)]);
}

#[test]
fn contains_reports_an_index_out_of_range() {
    setup();
    let m = matrix::<bool>(2, 2, &[(1, 0, true)]);
    assert_eq!(m.contains(1, 0), Ok(true));
    assert_eq!(m.contains(0, 0), Ok(false));
    assert_eq!(m.contains(2, 0).unwrap_err().kind, ErrorKind::InvalidIndex);
}

#[test]
fn a_failed_read_carries_no_message_from_an_earlier_call() {
    setup();
    let mut m = Matrix::<f64>::new(2, 2).unwrap();
    let written = m.set_element(2, 0, 1.0).unwrap_err();
    assert!(
        written.message.contains("Row index 2 out of range"),
        "{written}"
    );
    // GraphBLAS logs into an object only when it is the output, so that
    // text is still attached to `m`. A read must not report it as its own.
    let read = m.extract_element(0, 9).unwrap_err();
    assert_eq!(read.kind, ErrorKind::InvalidIndex);
    assert_eq!(read.message, "");
    assert_eq!(m.contains(0, 9).unwrap_err().message, "");
}

#[test]
fn transpose_with_no_descriptor_transposes() {
    setup();
    let a = matrix::<u64>(2, 3, &[(0, 2, 9), (1, 0, 4)]);
    let mut t = Matrix::<u64>::new(3, 2).unwrap();
    matrix::transpose(&mut t, Mask::NONE, None, &a, Desc::NONE).unwrap();
    assert_eq!(entries(&t), [(0, 1, 4), (2, 0, 9)]);
}

#[test]
fn structural_and_valued_masks_differ_on_a_stored_false() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1u64), (1, 1, 2), (0, 1, 3)]);
    let m = matrix(2, 2, &[(0, 0, true), (1, 1, false)]);

    // `C<M> = A` is `transpose` with `T0`; the GraphBLAS idiom for a masked copy.
    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::structural(&m), None, &a, Desc::T0).unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 1), (1, 1, 2)],
        "structural: both entries"
    );

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::valued(&m), None, &a, Desc::T0).unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 1)],
        "valued: the stored false is out"
    );

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::structural_complement(&m), None, &a, Desc::T0).unwrap();
    assert_eq!(entries(&c), vec![(0, 1, 3)]);

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::transpose(&mut c, Mask::valued_complement(&m), None, &a, Desc::T0).unwrap();
    assert_eq!(entries(&c), vec![(0, 1, 3), (1, 1, 2)]);
}

#[test]
fn replace_clears_outside_the_mask() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1u64)]);
    let m = matrix(2, 2, &[(0, 0, true)]);
    let mut c = matrix(2, 2, &[(1, 1, 9u64), (0, 0, 5)]);
    matrix::transpose(&mut c, Mask::structural(&m), None, &a, Desc::T0.replace()).unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 1)],
        "replace drops (1,1), which the mask excludes"
    );

    let mut c = matrix(2, 2, &[(1, 1, 9u64), (0, 0, 5)]);
    matrix::transpose(&mut c, Mask::structural(&m), None, &a, Desc::T0).unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 1), (1, 1, 9)],
        "without replace it stays"
    );
}

#[test]
fn output_as_a_structural_mask() {
    setup();
    // C<C> = A*B: keep only the positions C already has.
    let a = matrix(2, 2, &[(0, 0, true), (0, 1, true), (1, 1, true)]);
    let mut c = matrix(2, 2, &[(0, 0, false)]);
    mxm(
        &mut c,
        Mask::structural(Output),
        None,
        &ANY_PAIR_BOOL,
        &a,
        &a,
        Desc::REPLACE,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 0, true)]);
}

#[test]
fn descriptor_table_covers_all_32_combinations() {
    setup();
    let modes = [
        (MaskMode::NONE, ""),
        (
            MaskMode {
                structural: true,
                complement: false,
            },
            "S",
        ),
        (
            MaskMode {
                structural: false,
                complement: true,
            },
            "C",
        ),
        (
            MaskMode {
                structural: true,
                complement: true,
            },
            "SC",
        ),
    ];
    let mut seen = std::collections::HashSet::new();
    for (mode, mode_name) in modes {
        for bits in 0..8u8 {
            let desc = Desc {
                replace: bits & 1 != 0,
                transpose_first: bits & 2 != 0,
                transpose_second: bits & 4 != 0,
            };
            let got = descriptor(mode, desc);
            let mut expected_name = String::from("GrB_DESC_");
            if desc.replace {
                expected_name.push('R');
            }
            expected_name.push_str(mode_name);
            if desc.transpose_first {
                expected_name.push_str("T0");
            }
            if desc.transpose_second {
                expected_name.push_str("T1");
            }
            if expected_name == "GrB_DESC_" {
                assert!(got.is_null(), "all-default is the null descriptor");
                continue;
            }
            assert!(!got.is_null(), "{expected_name}");
            assert!(
                seen.insert(got as usize),
                "{expected_name} handed out twice"
            );
            let name = handle_name(got);
            assert_eq!(name, expected_name);
        }
    }
    assert_eq!(seen.len(), 31);
}

/// The name GraphBLAS reports for a built-in descriptor.
fn handle_name(d: sys::GrB_Descriptor) -> String {
    let mut buf = [0 as c_char; sys::GxB_MAX_NAME_LEN as usize];
    // SAFETY: `d` is a live built-in descriptor and the buffer has the size
    // GraphBLAS requires for a name.
    let info = unsafe {
        sys::GrB_Descriptor_get_String(d, buf.as_mut_ptr(), sys::GxB_Option_Field::GrB_NAME as _)
    };
    assert_eq!(info, GrB_Info::GrB_SUCCESS);
    // SAFETY: GraphBLAS wrote a NUL-terminated string into `buf`.
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

// ───────────────────────────── operations ─────────────────────────────

#[test]
fn accumulator_is_spelled_at_the_call() {
    setup();
    let a = matrix(1, 1, &[(0, 0, 2u64)]);
    let mut c = matrix(1, 1, &[(0, 0, 1u64)]);
    matrix::apply(&mut c, Mask::NONE, None, &IDENTITY_UINT64, &a, Desc::NONE).unwrap();
    assert_eq!(
        c.extract_element(0, 0).unwrap(),
        Some(2),
        "no accum: overwritten"
    );

    let mut c = matrix(1, 1, &[(0, 0, 1u64)]);
    matrix::apply(
        &mut c,
        Mask::NONE,
        Some(&PLUS_UINT64),
        &IDENTITY_UINT64,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(c.extract_element(0, 0).unwrap(), Some(3), "accum: added");
}

#[test]
fn pattern_semiring_accepts_any_input_type_and_reads_no_value() {
    setup();
    // Ids of 0 would become `false` under GraphBLAS's cast to bool; a
    // `Pattern` semiring never looks.
    let a = matrix(2, 2, &[(0, 0, 0u64), (0, 1, 0)]);
    let b = matrix(2, 2, &[(0, 0, 0.0f64), (1, 0, 0.0)]);
    let mut c = Matrix::<bool>::new(2, 2).unwrap();
    mxm(&mut c, Mask::NONE, None, &ANY_PAIR_BOOL, &a, &b, Desc::NONE).unwrap();
    assert_eq!(entries(&c), vec![(0, 0, true)]);

    let mut n = Matrix::<u64>::new(2, 2).unwrap();
    mxm(
        &mut n,
        Mask::NONE,
        None,
        &PLUS_PAIR_UINT64,
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(
        entries(&n),
        vec![(0, 0, 2)],
        "two paths 0->0->0 and 0->1->0"
    );
}

#[test]
fn mxm_over_f64_with_transposes() {
    setup();
    let a = matrix(2, 3, &[(0, 0, 1.0f64), (0, 2, 2.0), (1, 1, 3.0)]);
    let b = matrix(2, 3, &[(0, 2, 4.0f64), (1, 1, 5.0)]);
    // C = A * Bᵀ : 2x2
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    mxm(
        &mut c,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64,
        &a,
        &b,
        Desc::T1,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 0, 8.0), (1, 1, 15.0)]);
    // C = Aᵀ * B : 3x3
    let mut c = Matrix::<f64>::new(3, 3).unwrap();
    mxm(
        &mut c,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64,
        &a,
        &b,
        Desc::T0,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 2, 4.0), (1, 1, 15.0), (2, 2, 8.0)]);
}

#[test]
fn accumulate_a_product_with_the_semirings_add_op() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1.0f64), (1, 1, 2.0)]);
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    matrix::assign_scalar(
        &mut c,
        Mask::NONE,
        None,
        10.0,
        Indices::All,
        Indices::All,
        Desc::NONE,
    )
    .unwrap();
    let s = &PLUS_TIMES_SEMIRING_FP64;
    assert_eq!(
        s.add_op().as_raw(),
        PLUS_FP64.as_raw(),
        "add_op is the monoid's op"
    );
    mxm(&mut c, Mask::NONE, Some(&s.add_op()), s, &a, &a, Desc::NONE).unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 11.0), (0, 1, 10.0), (1, 0, 10.0), (1, 1, 14.0)]
    );
}

#[test]
fn apply_in_place_through_output() {
    setup();
    let mut c = matrix(2, 2, &[(0, 0, 1.0f64), (1, 1, 2.0)]);
    let before = c.as_raw();
    matrix::apply(&mut c, Mask::NONE, None, &AINV_FP64, Output, Desc::NONE).unwrap();
    assert_eq!(c.as_raw(), before, "the handle is the same object");
    assert_eq!(entries(&c), vec![(0, 0, -1.0), (1, 1, -2.0)]);
}

#[test]
fn mxm_with_output_as_an_input() {
    setup();
    let mut c = matrix(2, 2, &[(0, 1, true), (1, 0, true)]);
    mxm(
        &mut c,
        Mask::NONE,
        None,
        &ANY_PAIR_BOOL,
        Output,
        Output,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 0, true), (1, 1, true)], "C = C*C");
}

#[test]
fn ewise_add_is_union_and_ewise_mult_is_intersection() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1u64), (0, 1, 2)]);
    let b = matrix(2, 2, &[(0, 1, 10u64), (1, 1, 20)]);
    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::ewise_add(&mut c, Mask::NONE, None, &PLUS_UINT64, &a, &b, Desc::NONE).unwrap();
    assert_eq!(entries(&c), vec![(0, 0, 1), (0, 1, 12), (1, 1, 20)]);

    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::ewise_mult(&mut c, Mask::NONE, None, &PLUS_UINT64, &a, &b, Desc::NONE).unwrap();
    assert_eq!(entries(&c), vec![(0, 1, 12)]);

    // A monoid's operator is passed as such.
    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::ewise_add(
        &mut c,
        Mask::NONE,
        None,
        &PLUS_MONOID_UINT64.op(),
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(c.nvals(), 3);

    // The output type of ewise_mult is the operator's.
    let mut gt = Matrix::<bool>::new(2, 2).unwrap();
    matrix::ewise_mult(&mut gt, Mask::NONE, None, &GT_UINT64, &a, &b, Desc::NONE).unwrap();
    assert_eq!(entries(&gt), vec![(0, 1, false)]);
}

#[test]
fn select_keeps_values_and_apply_index_sees_positions() {
    setup();
    let a = matrix(3, 3, &[(0, 0, 1.0f64), (1, 2, 2.0), (2, 2, 3.0)]);
    let mut c = Matrix::<f64>::new(3, 3).unwrap();
    matrix::select(&mut c, Mask::NONE, None, &VALUEGT_FP64, &a, 1.5, Desc::NONE).unwrap();
    assert_eq!(entries(&c), vec![(1, 2, 2.0), (2, 2, 3.0)]);

    let mut d = Matrix::<f64>::new(3, 3).unwrap();
    matrix::select(&mut d, Mask::NONE, None, &DIAG, &a, 0, Desc::NONE).unwrap();
    assert_eq!(entries(&d), vec![(0, 0, 1.0), (2, 2, 3.0)]);

    let mut off = Matrix::<f64>::new(3, 3).unwrap();
    matrix::select(&mut off, Mask::NONE, None, &OFFDIAG, Output, 0, Desc::NONE).unwrap();
    assert_eq!(off.nvals(), 0, "select on an empty Output");

    let mut rows = Matrix::<i64>::new(3, 3).unwrap();
    matrix::apply_index(
        &mut rows,
        Mask::NONE,
        None,
        &ROWINDEX_INT64,
        &a,
        0,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&rows), vec![(0, 0, 0), (1, 2, 1), (2, 2, 2)]);
}

#[test]
fn assign_and_extract_sub_grids() {
    setup();
    let mut c = Matrix::<bool>::new(3, 3).unwrap();
    matrix::assign_scalar(
        &mut c,
        Mask::NONE,
        None,
        true,
        Indices::List(&[0, 2]),
        Indices::All,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(c.nvals(), 6);
    assert!(c.contains(2, 1).unwrap() && !c.contains(1, 1).unwrap());

    let a = matrix(2, 2, &[(0, 0, false), (1, 1, false)]);
    matrix::assign(
        &mut c,
        Mask::NONE,
        None,
        &a,
        Indices::List(&[1, 2]),
        Indices::List(&[0, 1]),
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(c.extract_element(1, 0).unwrap(), Some(false));
    assert_eq!(c.extract_element(2, 1).unwrap(), Some(false));
    assert!(
        !c.contains(2, 0).unwrap(),
        "C(I,J) = A deletes what A lacks"
    );

    let mut sub = Matrix::<bool>::new(1, 3).unwrap();
    matrix::extract(
        &mut sub,
        Mask::NONE,
        None,
        &c,
        Indices::List(&[2]),
        Indices::All,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&sub), vec![(0, 1, false), (0, 2, true)]);
}

#[test]
fn reduce_to_scalar_and_to_vector() {
    setup();
    let a = matrix(2, 3, &[(0, 0, 1u64), (0, 2, 2), (1, 1, 3)]);
    let mut total = 100u64;
    matrix::reduce(&mut total, None, &PLUS_MONOID_UINT64, &a, Desc::NONE).unwrap();
    assert_eq!(total, 6, "no accum overwrites");
    matrix::reduce(
        &mut total,
        Some(&PLUS_UINT64),
        &PLUS_MONOID_UINT64,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(total, 12);

    let empty = Matrix::<u64>::new(2, 2).unwrap();
    let mut x = 7u64;
    matrix::reduce(&mut x, None, &PLUS_MONOID_UINT64, &empty, Desc::NONE).unwrap();
    assert_eq!(x, 0, "an empty matrix reduces to the identity");
    assert_eq!(PLUS_MONOID_UINT64.identity(), 0);
}

#[test]
fn cast_changes_the_element_type_on_purpose() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 0u64), (1, 1, 5)]);
    let mut b = Matrix::<bool>::new(2, 2).unwrap();
    matrix::cast(&mut b, Mask::NONE, None, &a, Desc::NONE).unwrap();
    assert_eq!(
        entries(&b),
        vec![(0, 0, false), (1, 1, true)],
        "GraphBLAS's cast: 0 is false"
    );
    let mut f = Matrix::<f64>::new(2, 2).unwrap();
    matrix::cast(&mut f, Mask::NONE, None, &a, Desc::NONE).unwrap();
    assert_eq!(entries(&f), vec![(0, 0, 0.0), (1, 1, 5.0)]);
}

#[test]
fn errors_carry_the_code_and_graphblas_message() {
    setup();
    let a = Matrix::<f64>::new(2, 3).unwrap();
    let mut c = Matrix::<f64>::new(2, 2).unwrap();
    let err = mxm(
        &mut c,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64,
        &a,
        &a,
        Desc::NONE,
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::DimensionMismatch);
    assert!(err.message.contains("Dimensions not compatible"), "{err}");
    assert!(
        err.to_string()
            .starts_with("GraphBLAS: GrB_DIMENSION_MISMATCH")
    );
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

unsafe extern "C" fn double_u64(
    z: *mut u64,
    x: *const u64,
) {
    // SAFETY: as above.
    unsafe { *z = *x * 2 };
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
    let mut m = Matrix::<Edge>::new(2, 2).unwrap();
    m.set_element(0, 1, e).unwrap();
    assert_eq!(m.extract_element(0, 1).unwrap(), Some(e));
    assert_eq!(m.extract_element(1, 1).unwrap(), None);
    let (_, _, vals) = m.extract_tuples().unwrap();
    assert_eq!(vals, vec![e]);

    let mut s = Scalar::<Edge>::new().unwrap();
    s.set(e).unwrap();
    assert_eq!(s.get().unwrap(), Some(e));
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
    let above = &ID_ABOVE;
    assert_eq!(edge_max.op().as_raw(), heavier_op.as_raw());
    assert_eq!(edge_max.identity().id, 0);
    assert_eq!(heaviest_path.add().as_raw(), edge_max.as_raw());
    assert_eq!(heaviest_path.multiply().as_raw(), heavier_op.as_raw());
    assert_eq!(
        EDGE_WEIGHT.as_raw(),
        weight.as_raw(),
        "a static is created once"
    );

    let a = matrix(
        2,
        2,
        &[
            (0, 0, Edge { weight: 1.0, id: 1 }),
            (0, 1, Edge { weight: 3.0, id: 2 }),
        ],
    );
    let mut w = Matrix::<f64>::new(2, 2).unwrap();
    matrix::apply(&mut w, Mask::NONE, None, weight, &a, Desc::NONE).unwrap();
    assert_eq!(entries(&w), vec![(0, 0, 1.0), (0, 1, 3.0)]);

    let mut c = Matrix::<Edge>::new(2, 2).unwrap();
    mxm(&mut c, Mask::NONE, None, heaviest_path, &a, &a, Desc::NONE).unwrap();
    assert_eq!(c.extract_element(0, 0).unwrap().map(|e| e.id), Some(1));
    assert_eq!(c.extract_element(0, 1).unwrap().map(|e| e.id), Some(2));

    let mut heaviest = Edge { weight: 0.0, id: 0 };
    matrix::reduce(&mut heaviest, None, edge_max, &a, Desc::NONE).unwrap();
    assert_eq!(heaviest.id, 2);

    let mut sel = Matrix::<Edge>::new(2, 2).unwrap();
    matrix::select(&mut sel, Mask::NONE, None, above, &a, 1, Desc::NONE).unwrap();
    assert_eq!(sel.nvals(), 1);

    // An operator built ad hoc is registered too; it just has no static.
    // SAFETY: `double_u64` reads `x` before it writes `z`.
    let twice = unsafe { UnaryOp::<u64, u64>::new(double_u64) }.unwrap();
    let mut n = matrix(1, 1, &[(0, 0, 21u64)]);
    matrix::apply(&mut n, Mask::NONE, None, &twice, Output, Desc::NONE).unwrap();
    assert_eq!(n.extract_element(0, 0).unwrap(), Some(42));
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
