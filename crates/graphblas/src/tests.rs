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
    sync::{Barrier, Once, mpsc},
    thread,
};

use crate::{
    AsBool, Desc, Error, ErrorKind, Frozen, Indices, JitControl, Lazy, Mask, Matrix, Output,
    Scalar, Sparsity, Vector,
    init::{init, is_initialized, set_jit},
    mask::{MaskMode, descriptor},
    matrix, mxm, mxv,
    ops::*,
    sys::{self, GrB_Index, GrB_Info},
    types::{GrbType, Pattern},
    vector, vxm,
};

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

fn vector<T: GrbType>(
    size: GrB_Index,
    entries: &[(GrB_Index, T)],
) -> Vector<T> {
    let mut v = Vector::new(size).unwrap();
    for &(i, x) in entries {
        v.set_element(i, x).unwrap();
    }
    v
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

fn vector_entries<T: GrbType + PartialOrd>(v: &Vector<T>) -> Vec<(GrB_Index, T)> {
    let (idx, vals) = v.extract_tuples().unwrap();
    let mut out: Vec<_> = idx.into_iter().zip(vals).collect();
    out.sort_by_key(|e| e.0);
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

#[test]
fn vector_handle_basics() {
    setup();
    let mut v = vector(5, &[(1, 10u64), (3, 30)]);
    assert_eq!((v.size(), v.nvals()), (5, 2));
    assert_eq!(v.extract_element(3).unwrap(), Some(30));
    assert!(v.contains(1).unwrap() && !v.contains(0).unwrap());
    v.remove_element(1).unwrap();
    assert_eq!(vector_entries(&v), vec![(3, 30)]);
    let mut w = Vector::<u64>::new(3).unwrap();
    w.build(&[2, 0], &[2, 0], None).unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 0), (2, 2)]);
    let mut iso = Vector::<bool>::new(3).unwrap();
    iso.build_iso(&[0, 2], true).unwrap();
    assert!(iso.is_iso());
    assert_eq!(
        format!("{v:?}").split(" @ ").next(),
        Some("Vector<GrB_UINT64>(5")
    );
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

    let v = vector::<u64>(2, &[(1, 3)]);
    let u = vector::<u64>(2, &[(0, 1), (1, 2)]);
    let mut w = Vector::<u64>::new(2).unwrap();
    vector::apply(
        &mut w,
        Mask::valued(AsBool(&v)),
        None,
        &IDENTITY_UINT64,
        &u,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(1, 2)]);
}

#[test]
fn contains_reports_an_index_out_of_range() {
    setup();
    let m = matrix::<bool>(2, 2, &[(1, 0, true)]);
    assert_eq!(m.contains(1, 0), Ok(true));
    assert_eq!(m.contains(0, 0), Ok(false));
    assert_eq!(m.contains(2, 0).unwrap_err().kind, ErrorKind::InvalidIndex);
    let v = vector::<bool>(2, &[(1, true)]);
    assert_eq!(v.contains(5).unwrap_err().kind, ErrorKind::InvalidIndex);
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

    let mut v = Vector::<f64>::new(2).unwrap();
    assert_ne!(v.set_element(5, 1.0).unwrap_err().message, "");
    assert_eq!(v.extract_element(9).unwrap_err().message, "");
    assert_eq!(v.contains(9).unwrap_err().message, "");
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
fn vxm_and_mxv_transpose_the_matrix_by_its_position() {
    setup();
    // One edge 0 -> 1, and a vector holding node 0.
    let a = matrix::<bool>(2, 2, &[(0, 1, true)]);
    let u = vector::<bool>(2, &[(0, true)]);
    let row_times_matrix = |desc| {
        let mut w = Vector::<bool>::new(2).unwrap();
        vxm(&mut w, Mask::NONE, None, &ANY_PAIR_BOOL, &u, &a, desc).unwrap();
        vector_entries(&w)
    };
    let matrix_times_column = |desc| {
        let mut w = Vector::<bool>::new(2).unwrap();
        mxv(&mut w, Mask::NONE, None, &ANY_PAIR_BOOL, &a, &u, desc).unwrap();
        vector_entries(&w)
    };
    let (followed, nothing) = (vec![(1, true)], vec![]);

    // u * A follows the edge. The matrix is the second input.
    assert_eq!(row_times_matrix(Desc::NONE), followed);
    assert_eq!(row_times_matrix(Desc::T1), nothing);
    assert_eq!(row_times_matrix(Desc::T0), followed, "T0 names the vector");
    // A * u goes against the edge. The matrix is the first input.
    assert_eq!(matrix_times_column(Desc::NONE), nothing);
    assert_eq!(matrix_times_column(Desc::T0), followed);
    assert_eq!(
        matrix_times_column(Desc::T1),
        nothing,
        "T1 names the vector"
    );
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

    let mut rows = Vector::<u64>::new(2).unwrap();
    matrix::reduce_rows(
        &mut rows,
        Mask::NONE,
        None,
        &PLUS_MONOID_UINT64,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&rows), vec![(0, 3), (1, 3)]);
    let mut cols = Vector::<u64>::new(3).unwrap();
    matrix::reduce_rows(
        &mut cols,
        Mask::NONE,
        None,
        &PLUS_MONOID_UINT64,
        &a,
        Desc::T0,
    )
    .unwrap();
    assert_eq!(vector_entries(&cols), vec![(0, 1), (1, 3), (2, 2)]);
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
fn vector_operations_mirror_the_matrix_ones() {
    setup();
    // One BFS step over a 3-node path 0 -> 1 -> 2.
    let a = matrix(3, 3, &[(0, 1, true), (1, 2, true)]);
    let frontier = vector(3, &[(0, true)]);
    let mut next = Vector::<bool>::new(3).unwrap();
    vxm(
        &mut next,
        Mask::NONE,
        None,
        &ANY_PAIR_BOOL,
        &frontier,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&next), vec![(1, true)]);
    let mut back = Vector::<bool>::new(3).unwrap();
    mxv(
        &mut back,
        Mask::NONE,
        None,
        &ANY_PAIR_BOOL,
        &a,
        &next,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&back), vec![(0, true)]);

    // vxm with Output as the vector input, masked by the complement of the
    // visited set: the frontier advances in place.
    let mut f = vector(3, &[(0, true)]);
    let visited = vector(3, &[(0, true)]);
    vxm(
        &mut f,
        Mask::structural_complement(&visited),
        None,
        &ANY_PAIR_BOOL,
        Output,
        &a,
        Desc::REPLACE,
    )
    .unwrap();
    assert_eq!(vector_entries(&f), vec![(1, true)]);

    let u = vector(3, &[(0, 1u64), (1, 2)]);
    let v = vector(3, &[(1, 10u64), (2, 20)]);
    let mut w = Vector::<u64>::new(3).unwrap();
    vector::ewise_add(&mut w, Mask::NONE, None, &PLUS_UINT64, &u, &v, Desc::NONE).unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 1), (1, 12), (2, 20)]);
    vector::ewise_mult(&mut w, Mask::NONE, None, &PLUS_UINT64, &u, &v, Desc::NONE).unwrap();
    assert_eq!(vector_entries(&w), vec![(1, 12)]);
    vector::apply(&mut w, Mask::NONE, None, &IDENTITY_UINT64, &u, Desc::NONE).unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 1), (1, 2)]);
    vector::select(
        &mut w,
        Mask::NONE,
        None,
        &VALUEEQ_UINT64,
        Output,
        2,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(1, 2)]);
    vector::assign_scalar(
        &mut w,
        Mask::NONE,
        None,
        5,
        Indices::List(&[0, 2]),
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 5), (1, 2), (2, 5)]);
    let mut sub = Vector::<u64>::new(2).unwrap();
    vector::extract(
        &mut sub,
        Mask::NONE,
        None,
        &w,
        Indices::List(&[2, 1]),
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&sub), vec![(0, 5), (1, 2)]);
    vector::assign(
        &mut w,
        Mask::NONE,
        None,
        &sub,
        Indices::List(&[1, 2]),
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 5), (1, 5), (2, 2)]);
    let mut total = 0u64;
    vector::reduce(&mut total, None, &PLUS_MONOID_UINT64, &w, Desc::NONE).unwrap();
    assert_eq!(total, 12);
    let mut b = Vector::<bool>::new(3).unwrap();
    vector::cast(&mut b, Mask::NONE, None, &w, Desc::NONE).unwrap();
    assert_eq!(b.nvals(), 3);
    let mut r = Vector::<i64>::new(3).unwrap();
    vector::apply_index(&mut r, Mask::NONE, None, &ROWINDEX_INT64, &w, 0, Desc::NONE).unwrap();
    assert_eq!(vector_entries(&r), vec![(0, 0), (1, 1), (2, 2)]);
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

    let mut v = Vector::<Edge>::new(2).unwrap();
    v.build(&[1], &[e], None).unwrap();
    assert_eq!(v.extract_element(1).unwrap(), Some(e));
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

// ───────────────────────────── typecast ─────────────────────────────

#[test]
fn typecast_relabels_the_same_handle() {
    setup();
    assert_eq!(
        PLUS_TIMES_SEMIRING_FP64
            .typecast::<u64, bool, u64>()
            .as_raw(),
        PLUS_TIMES_SEMIRING_FP64.as_raw()
    );
    assert_eq!(
        PLUS_UINT64.typecast::<bool, bool, f64>().as_raw(),
        PLUS_UINT64.as_raw()
    );
    assert_eq!(
        IDENTITY_FP64.typecast::<i64, i64>().as_raw(),
        IDENTITY_FP64.as_raw()
    );
    assert_eq!(
        VALUEGT_FP64.typecast::<u64, bool>().as_raw(),
        VALUEGT_FP64.as_raw()
    );
    assert_eq!(
        PLUS_MONOID_UINT64.typecast::<bool, f64>().as_raw(),
        PLUS_MONOID_UINT64.as_raw()
    );
    // The parts of a relabelled semiring are its own, relabelled alike.
    let semiring = PLUS_TIMES_SEMIRING_FP64.typecast::<u64, bool, u64>();
    assert_eq!(
        semiring.multiply().as_raw(),
        PLUS_TIMES_SEMIRING_FP64.multiply().as_raw()
    );
}

#[test]
fn a_typecast_semiring_runs_on_operands_of_other_types() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 2u64), (0, 1, 3)]);
    let b = matrix(2, 2, &[(0, 0, true), (1, 0, true), (1, 1, false)]);

    // Domains inferred from the operands: u64 x bool -> u64 through f64.
    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    mxm(
        &mut c,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64.typecast(),
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(
        entries(&c),
        vec![(0, 0, 5), (0, 1, 0)],
        "true is 1.0 and false is 0.0; the entry at (0, 1) is a stored zero"
    );

    // The same call with the domains written out, against the result of
    // storing converted copies and running the semiring as declared.
    let mut a_f64 = Matrix::<f64>::new(2, 2).unwrap();
    matrix::cast(&mut a_f64, Mask::NONE, None, &a, Desc::NONE).unwrap();
    let mut b_f64 = Matrix::<f64>::new(2, 2).unwrap();
    matrix::cast(&mut b_f64, Mask::NONE, None, &b, Desc::NONE).unwrap();
    let mut product = Matrix::<f64>::new(2, 2).unwrap();
    mxm(
        &mut product,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64,
        &a_f64,
        &b_f64,
        Desc::NONE,
    )
    .unwrap();
    let mut expected = Matrix::<u64>::new(2, 2).unwrap();
    matrix::cast(&mut expected, Mask::NONE, None, &product, Desc::NONE).unwrap();

    let spelled = PLUS_TIMES_SEMIRING_FP64.typecast::<u64, bool, u64>();
    let mut d = Matrix::<u64>::new(2, 2).unwrap();
    mxm(&mut d, Mask::NONE, None, &spelled, &a, &b, Desc::NONE).unwrap();
    assert_eq!(entries(&d), entries(&expected));

    // With an accumulator the product is cast to the output's type and
    // accumulated there.
    mxm(
        &mut d,
        Mask::NONE,
        Some(&PLUS_UINT64),
        &spelled,
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&d), vec![(0, 0, 10), (0, 1, 0)]);
}

#[test]
fn a_typecast_semiring_adds_in_its_own_type_and_casts_the_sum_once() {
    setup();
    // 0.5 * 1 + -0.5 * 1 is 0.0, which is `false`. A semiring over bool
    // would have cast each input first and found a path.
    let a = matrix(1, 2, &[(0, 0, 0.5f64), (0, 1, -0.5)]);
    let b = matrix(2, 1, &[(0, 0, 1.0f64), (1, 0, 1.0)]);
    let mut reachable = Matrix::<bool>::new(1, 1).unwrap();
    mxm(
        &mut reachable,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64.typecast(),
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&reachable), vec![(0, 0, false)]);

    mxm(
        &mut reachable,
        Mask::NONE,
        None,
        &LOR_LAND_SEMIRING_BOOL.typecast(),
        &a,
        &b,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&reachable), vec![(0, 0, true)]);
}

#[test]
fn a_positional_semiring_writes_indices_into_an_unsigned_vector() {
    setup();
    // The idiom of the workspace notes: each column of `a` has one entry,
    // and FIRSTJ gives the index into `x`, the entry's row, as an i64.
    let a = matrix(3, 3, &[(2, 0, true), (0, 1, true), (2, 2, true)]);
    let x = vector(3, &[(0, 0u8), (1, 0), (2, 0)]);
    let mut rows = Vector::<u64>::new(3).unwrap();
    vxm(
        &mut rows,
        Mask::NONE,
        None,
        &PLUS_FIRSTJ_INT64.typecast(),
        &x,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&rows), vec![(0, 2), (1, 0), (2, 2)]);

    let mut signed = Vector::<i64>::new(3).unwrap();
    vxm(
        &mut signed,
        Mask::NONE,
        None,
        &PLUS_FIRSTJ_INT64,
        &x,
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&signed), vec![(0, 2), (1, 0), (2, 2)]);

    // mxv, with the matrix in the first slot.
    let u = vector(3, &[(0, 2u64)]);
    let mut w = Vector::<i32>::new(3).unwrap();
    mxv(
        &mut w,
        Mask::NONE,
        None,
        &PLUS_TIMES_SEMIRING_FP64.typecast(),
        &a,
        &u,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(2, 2)]);
}

#[test]
fn typecast_converts_as_rust_as_does_and_to_bool_as_nonzero() {
    setup();
    let floats = matrix(
        1,
        6,
        &[
            (0, 0, f64::NAN),
            (0, 1, 1e300),
            (0, 2, -1e300),
            (0, 3, 2.9),
            (0, 4, -2.9),
            (0, 5, 0.0),
        ],
    );
    let (_, _, values) = floats.extract_tuples().unwrap();

    let mut signed = Matrix::<i64>::new(1, 6).unwrap();
    let to_i64 = IDENTITY_FP64.typecast::<f64, i64>();
    matrix::apply(&mut signed, Mask::NONE, None, &to_i64, &floats, Desc::NONE).unwrap();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the truncation is what the test compares against"
    )]
    let expected: Vec<i64> = values.iter().map(|&x| x as i64).collect();
    assert_eq!(expected, vec![0, i64::MAX, i64::MIN, 2, -2, 0]);
    let (_, _, got) = signed.extract_tuples().unwrap();
    assert_eq!(got, expected, "NaN is 0 and the range saturates");

    let mut unsigned = Matrix::<u8>::new(1, 6).unwrap();
    matrix::apply(
        &mut unsigned,
        Mask::NONE,
        None,
        &IDENTITY_FP64.typecast(),
        &floats,
        Desc::NONE,
    )
    .unwrap();
    let (_, _, got) = unsigned.extract_tuples().unwrap();
    assert_eq!(got, vec![0, u8::MAX, 0, 2, 0, 0]);

    let mut flags = Matrix::<bool>::new(1, 6).unwrap();
    matrix::apply(
        &mut flags,
        Mask::NONE,
        None,
        &IDENTITY_FP64.typecast(),
        &floats,
        Desc::NONE,
    )
    .unwrap();
    let (_, _, got) = flags.extract_tuples().unwrap();
    assert_eq!(
        got,
        vec![true, true, true, true, true, false],
        "anything but zero, NaN included"
    );

    // The same conversions on the way in: each float is cast to the
    // operator's i64 before `PLUS_INT64` sees it.
    let zero_cells: Vec<_> = (0..6).map(|j| (0, j, 0.0f64)).collect();
    let zeros = matrix(1, 6, &zero_cells);
    let mut sums = Matrix::<i64>::new(1, 6).unwrap();
    matrix::ewise_mult(
        &mut sums,
        Mask::NONE,
        None,
        &PLUS_INT64.typecast(),
        &floats,
        &zeros,
        Desc::NONE,
    )
    .unwrap();
    let (_, _, got) = sums.extract_tuples().unwrap();
    assert_eq!(got, expected, "cast on the way in");

    // Integer to a narrower integer wraps.
    let wide = matrix(1, 2, &[(0, 0, 300u64), (0, 1, u64::MAX)]);
    let mut narrow = Matrix::<u8>::new(1, 2).unwrap();
    matrix::apply(
        &mut narrow,
        Mask::NONE,
        None,
        &IDENTITY_UINT64.typecast(),
        &wide,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&narrow), vec![(0, 0, 44), (0, 1, 255)]);
    let mut signed = Matrix::<i64>::new(1, 2).unwrap();
    matrix::apply(
        &mut signed,
        Mask::NONE,
        None,
        &IDENTITY_UINT64.typecast(),
        &wide,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&signed), vec![(0, 0, 300), (0, 1, -1)]);
}

#[test]
fn a_typecast_union_casts_the_entries_only_one_side_has() {
    setup();
    let flags = matrix(1, 3, &[(0, 0, true), (0, 1, true)]);
    let counts = matrix(1, 3, &[(0, 1, 5u64), (0, 2, 7)]);
    let mut total = Matrix::<u64>::new(1, 3).unwrap();
    matrix::ewise_add(
        &mut total,
        Mask::NONE,
        None,
        &PLUS_UINT64.typecast(),
        &flags,
        &counts,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(
        entries(&total),
        vec![(0, 0, 1), (0, 1, 6), (0, 2, 7)],
        "the lone `true` at column 0 became 1 without the operator"
    );

    // Into a bool output the lone entries are cast twice over: an id of 0
    // lands as `false`, which is what the note on `ewise_add` warns about.
    let ids = matrix(1, 3, &[(0, 2, 0u64)]);
    let mut seen = Matrix::<bool>::new(1, 3).unwrap();
    matrix::ewise_add(
        &mut seen,
        Mask::NONE,
        None,
        &ANY_BOOL.typecast(),
        &flags,
        &ids,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(
        entries(&seen),
        vec![(0, 0, true), (0, 1, true), (0, 2, false)]
    );

    // The intersection calls the operator on every entry it keeps.
    let mut both = Matrix::<f64>::new(1, 3).unwrap();
    matrix::ewise_mult(
        &mut both,
        Mask::NONE,
        None,
        &PLUS_UINT64.typecast(),
        &flags,
        &counts,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&both), vec![(0, 1, 6.0)]);

    // Vectors mirror the matrices.
    let u = vector(3, &[(0, true)]);
    let v = vector(3, &[(0, 2.5f64), (2, 1.5)]);
    let mut w = Vector::<f64>::new(3).unwrap();
    vector::ewise_add(
        &mut w,
        Mask::NONE,
        None,
        &PLUS_FP64.typecast(),
        &u,
        &v,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(0, 3.5), (2, 1.5)]);
    let mut x = Vector::<i64>::new(3).unwrap();
    vector::ewise_mult(
        &mut x,
        Mask::NONE,
        None,
        &PLUS_FP64.typecast(),
        &u,
        &v,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&x), vec![(0, 3)]);
    let mut y = Vector::<u8>::new(3).unwrap();
    vector::apply(
        &mut y,
        Mask::NONE,
        None,
        &AINV_FP64.typecast(),
        &v,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(
        vector_entries(&y),
        vec![(0, 0), (2, 0)],
        "-2.5 and -1.5 saturate at 0"
    );
}

#[test]
fn a_typecast_monoid_folds_entries_of_another_type() {
    setup();
    let flags = matrix(2, 2, &[(0, 0, true), (0, 1, true), (1, 1, false)]);
    let mut count = 0u64;
    matrix::reduce(
        &mut count,
        None,
        &PLUS_MONOID_UINT64.typecast(),
        &flags,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(count, 2, "the stored `false` adds 0");

    // Folded in u64 and cast once: 2 as f64, accumulated twice.
    let mut as_float = 0.0f64;
    let spelled = PLUS_MONOID_UINT64.typecast::<bool, f64>();
    matrix::reduce(
        &mut as_float,
        Some(&PLUS_FP64),
        &spelled,
        &flags,
        Desc::NONE,
    )
    .unwrap();
    matrix::reduce(
        &mut as_float,
        Some(&PLUS_FP64),
        &spelled,
        &flags,
        Desc::NONE,
    )
    .unwrap();
    assert!((as_float - 4.0).abs() < f64::EPSILON);

    let mut per_row = Vector::<u64>::new(2).unwrap();
    matrix::reduce_rows(
        &mut per_row,
        Mask::NONE,
        None,
        &PLUS_MONOID_UINT64.typecast(),
        &flags,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&per_row), vec![(0, 2), (1, 0)]);

    let v = vector(3, &[(0, 1.5f64), (2, 2.5)]);
    let mut truncated = 0u64;
    vector::reduce(
        &mut truncated,
        None,
        &PLUS_MONOID_UINT64.typecast(),
        &v,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(truncated, 3, "each entry is cast before the fold: 1 + 2");
    let mut summed = 0u64;
    vector::reduce(
        &mut summed,
        None,
        &PLUS_MONOID_FP64.typecast(),
        &v,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(summed, 4, "folded in f64 and cast once: 4.0");
}

#[test]
fn a_typecast_index_operator_writes_positions_into_another_type() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1u64), (0, 1, 5), (1, 1, 9)]);
    // Positions (i64 in GraphBLAS) into a u64 matrix.
    let mut columns = Matrix::<u64>::new(2, 2).unwrap();
    matrix::apply_index(
        &mut columns,
        Mask::NONE,
        None,
        &COLINDEX_INT64.typecast(),
        &a,
        10,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&columns), vec![(0, 0, 10), (0, 1, 11), (1, 1, 11)]);

    let u = vector(3, &[(0, 1u64), (2, 9)]);
    let mut at = Vector::<u32>::new(3).unwrap();
    vector::apply_index(
        &mut at,
        Mask::NONE,
        None,
        &ROWINDEX_INT64.typecast::<u64, u32>(),
        &u,
        1,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&at), vec![(0, 1), (2, 3)]);
}

/// `m` in every storage form GraphBLAS has: each takes its own kernels.
fn storage_forms<T: GrbType>(m: &Matrix<T>) -> Vec<Matrix<T>> {
    let mut forms = Vec::new();
    for orientation in [crate::Orientation::ByRow, crate::Orientation::ByCol] {
        for sparsity in [
            Sparsity::Hypersparse,
            Sparsity::Sparse,
            Sparsity::Bitmap,
            Sparsity::Full,
        ] {
            let mut form = m.dup().unwrap();
            form.set_orientation(orientation).unwrap();
            form.set_sparsity(sparsity).unwrap();
            form.wait(crate::WaitMode::Materialize).unwrap();
            forms.push(form);
        }
    }
    forms
}

/// The operators of one declared domain `D`, with the same arithmetic in
/// Rust: every value is held as an `f64` that `D` represents exactly.
struct Operators<D> {
    semiring: Semiring<D, D, D>,
    add_op: BinaryOp<D, D, D>,
    monoid: Monoid<D>,
    unary: UnaryOp<D, D>,
    /// GraphBLAS's cast of a value into `D`.
    declared: fn(f64) -> f64,
    add: fn(f64, f64) -> f64,
    multiply: fn(f64, f64) -> f64,
    apply: fn(f64) -> f64,
}

/// The typecast paths of GraphBLAS are its less travelled ones (`select`
/// and the union are wrong or inconsistent on one each, below), and which
/// kernel runs depends on the operator and on how the operands are stored.
/// So every operation that takes a typecast operator is run with operators
/// of three domains, on every storage form of its inputs, iso-valued and
/// not, against the same arithmetic done in Rust.
#[test]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the conversions are what the test compares GraphBLAS against, on small exact values"
)]
fn typecast_agrees_with_rust_conversions_in_every_storage_form() {
    setup();
    let mut forms_seen = Vec::new();
    // The float kernels are generic in the engine's build of GraphBLAS;
    // the integer and bool ones are its factory kernels.
    sweep(
        &Operators {
            semiring: *PLUS_TIMES_SEMIRING_FP64,
            add_op: *PLUS_FP64,
            monoid: *PLUS_MONOID_FP64,
            unary: *AINV_FP64,
            declared: |x| x,
            add: |x, y| x + y,
            multiply: |x, y| x * y,
            apply: |x| -x,
        },
        &mut forms_seen,
    );
    sweep(
        &Operators {
            semiring: *PLUS_TIMES_SEMIRING_UINT64,
            add_op: *PLUS_UINT64,
            monoid: *PLUS_MONOID_UINT64,
            unary: *IDENTITY_UINT64,
            declared: |x| (x as u64) as f64,
            add: |x, y| x + y,
            multiply: |x, y| x * y,
            apply: |x| x,
        },
        &mut forms_seen,
    );
    sweep(
        &Operators {
            semiring: *LOR_LAND_SEMIRING_BOOL,
            add_op: LOR_MONOID_BOOL.op(),
            monoid: *LOR_MONOID_BOOL,
            unary: *IDENTITY_BOOL,
            declared: |x| f64::from(u8::from(x != 0.0)),
            add: f64::max,
            multiply: f64::min,
            apply: |x| x,
        },
        &mut forms_seen,
    );
    for sparsity in [
        Sparsity::Hypersparse,
        Sparsity::Sparse,
        Sparsity::Bitmap,
        Sparsity::Full,
    ] {
        for iso in [false, true] {
            assert!(
                forms_seen
                    .iter()
                    .any(|&(s, _, i)| (s, i) == (sparsity, iso)),
                "no input was ever {sparsity:?} with iso = {iso}: {forms_seen:?}"
            );
        }
    }
}

type Form = (Sparsity, crate::Orientation, bool);

fn form<T: GrbType>(m: &Matrix<T>) -> Form {
    (m.sparsity(), m.orientation(), m.is_iso())
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the conversions are what the test compares GraphBLAS against, on small exact values"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one sweep over the operations, sharing its inputs and their forms"
)]
fn sweep<D: crate::BuiltinType>(
    domain: &Operators<D>,
    forms_seen: &mut Vec<Form>,
) {
    const N: GrB_Index = 5;
    let every_cell: Vec<(GrB_Index, GrB_Index)> =
        (0..N).flat_map(|i| (0..N).map(move |j| (i, j))).collect();
    let some_cells: Vec<(GrB_Index, GrB_Index)> = every_cell
        .iter()
        .copied()
        .filter(|&(i, j)| (i * 3 + j * 5) % 4 == 0)
        .collect();
    let fold = |terms: &[f64]| terms.iter().copied().reduce(domain.add);
    let weights = vector(N, &[(0, 2u8), (3, 1), (4, 5)]);
    let weight = |k| match k {
        0 => Some((domain.declared)(2.0)),
        3 => Some((domain.declared)(1.0)),
        4 => Some((domain.declared)(5.0)),
        _ => None,
    };

    for cells in [&some_cells, &every_cell] {
        let (rows, cols): (Vec<_>, Vec<_>) = cells.iter().copied().unzip();
        for (a_iso, b_iso) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut a = Matrix::<u64>::new(N, N).unwrap();
            let mut b = Matrix::<f32>::new(N, N).unwrap();
            if a_iso {
                a.build_iso(&rows, &cols, 3).unwrap();
            } else {
                let values: Vec<u64> = cells.iter().map(|&(i, j)| i * N + j + 1).collect();
                a.build(&rows, &cols, &values, None).unwrap();
            }
            if b_iso {
                b.build_iso(&rows, &cols, 1.5).unwrap();
            } else {
                // Halves: 0.5 is 0 as a u64 and `true` as a bool.
                let values: Vec<f32> = cells
                    .iter()
                    .map(|&(i, j)| (j * N + i + 1) as f32 * 0.5)
                    .collect();
                b.build(&rows, &cols, &values, None).unwrap();
            }
            assert_eq!((a.is_iso(), b.is_iso()), (a_iso, b_iso));

            // Each input as the operators see it, and the answers: the
            // arithmetic in the declared domain, one `as` at the end.
            let a_cells: Vec<_> = entries(&a)
                .into_iter()
                .map(|(i, j, x)| (i, j, (domain.declared)(x as f64)))
                .collect();
            let b_cells: Vec<_> = entries(&b)
                .into_iter()
                .map(|(i, j, x)| (i, j, (domain.declared)(f64::from(x))))
                .collect();
            let at = |cells: &[(GrB_Index, GrB_Index, f64)], i, j| {
                cells
                    .iter()
                    .find(|&&(row, col, _)| (row, col) == (i, j))
                    .map(|&(_, _, x)| x)
            };
            let mut product = Vec::new();
            let mut union = Vec::new();
            let mut intersection = Vec::new();
            for &(i, j) in &every_cell {
                let terms: Vec<f64> = (0..N)
                    .filter_map(|k| {
                        Some((domain.multiply)(at(&a_cells, i, k)?, at(&b_cells, k, j)?))
                    })
                    .collect();
                if let Some(sum) = fold(&terms) {
                    product.push((i, j, sum as i64));
                }
                match (at(&a_cells, i, j), at(&b_cells, i, j)) {
                    (Some(x), Some(y)) => {
                        union.push((i, j, (domain.add)(x, y) as i64));
                        intersection.push((i, j, (domain.add)(x, y) as i64));
                    }
                    (Some(x), None) | (None, Some(x)) => union.push((i, j, x as i64)),
                    (None, None) => {}
                }
            }
            let applied: Vec<_> = a_cells
                .iter()
                .map(|&(i, j, x)| (i, j, (domain.apply)(x) as i64))
                .collect();
            let row_numbers: Vec<_> = b_cells.iter().map(|&(i, j, _)| (i, j, i + 1)).collect();
            let all: Vec<f64> = b_cells.iter().map(|&(_, _, x)| x).collect();
            let total = fold(&all).unwrap() as i64;
            let row_totals: Vec<_> = (0..N)
                .filter_map(|i| {
                    let row: Vec<f64> = b_cells.iter().filter(|c| c.0 == i).map(|c| c.2).collect();
                    Some((i, fold(&row)? as i64))
                })
                .collect();
            let a_times_weights: Vec<_> = (0..N)
                .filter_map(|i| {
                    let terms: Vec<f64> = (0..N)
                        .filter_map(|k| Some((domain.multiply)(at(&a_cells, i, k)?, weight(k)?)))
                        .collect();
                    Some((i, fold(&terms)? as i64))
                })
                .collect();
            let weights_times_b: Vec<_> = (0..N)
                .filter_map(|j| {
                    let terms: Vec<f64> = (0..N)
                        .filter_map(|k| Some((domain.multiply)(weight(k)?, at(&b_cells, k, j)?)))
                        .collect();
                    Some((j, fold(&terms)? as i64))
                })
                .collect();

            let a_forms = storage_forms(&a);
            let b_forms = storage_forms(&b);
            let add_op = domain.add_op.typecast();
            let monoid = domain.monoid.typecast();
            let unary = domain.unary.typecast();
            let row_index = ROWINDEX_INT64.typecast();
            for form_a in &a_forms {
                if !forms_seen.contains(&form(form_a)) {
                    forms_seen.push(form(form_a));
                }
                let label = format!("{}, a {:?}", D::NAME, form(form_a));
                let mut c = Matrix::<i64>::new(N, N).unwrap();
                matrix::apply(&mut c, Mask::NONE, None, &unary, form_a, Desc::NONE).unwrap();
                assert_eq!(entries(&c), applied, "apply, {label}");
                let mut w = Vector::<i64>::new(N).unwrap();
                mxv(
                    &mut w,
                    Mask::NONE,
                    None,
                    &domain.semiring.typecast(),
                    form_a,
                    &weights,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(vector_entries(&w), a_times_weights, "mxv, {label}");
            }
            for form_b in &b_forms {
                let label = format!("{}, b {:?}", D::NAME, form(form_b));
                let mut positions = Matrix::<u64>::new(N, N).unwrap();
                matrix::apply_index(
                    &mut positions,
                    Mask::NONE,
                    None,
                    &row_index,
                    form_b,
                    1,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(entries(&positions), row_numbers, "apply_index, {label}");
                let mut sum = -1i64;
                matrix::reduce(&mut sum, None, &monoid, form_b, Desc::NONE).unwrap();
                assert_eq!(sum, total, "reduce, {label}");
                let mut sums = Vector::<i64>::new(N).unwrap();
                matrix::reduce_rows(&mut sums, Mask::NONE, None, &monoid, form_b, Desc::NONE)
                    .unwrap();
                assert_eq!(vector_entries(&sums), row_totals, "reduce_rows, {label}");
                let mut w = Vector::<i64>::new(N).unwrap();
                vxm(
                    &mut w,
                    Mask::NONE,
                    None,
                    &domain.semiring.typecast(),
                    &weights,
                    form_b,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(vector_entries(&w), weights_times_b, "vxm, {label}");
            }
            for form_a in &a_forms {
                for form_b in &b_forms {
                    let label = format!("{}, a {:?}, b {:?}", D::NAME, form(form_a), form(form_b));
                    let mut c = Matrix::<i64>::new(N, N).unwrap();
                    mxm(
                        &mut c,
                        Mask::NONE,
                        None,
                        &domain.semiring.typecast(),
                        form_a,
                        form_b,
                        Desc::NONE,
                    )
                    .unwrap();
                    assert_eq!(entries(&c), product, "mxm, {label}");
                    matrix::ewise_add(
                        &mut c,
                        Mask::NONE,
                        None,
                        &add_op,
                        form_a,
                        form_b,
                        Desc::NONE,
                    )
                    .unwrap();
                    assert_eq!(entries(&c), union, "ewise_add, {label}");
                    matrix::ewise_mult(
                        &mut c,
                        Mask::NONE,
                        None,
                        &add_op,
                        form_a,
                        form_b,
                        Desc::NONE,
                    )
                    .unwrap();
                    assert_eq!(entries(&c), intersection, "ewise_mult, {label}");
                }
            }
        }
    }
}

/// Why a typecast union needs an operator over one type: GraphBLAS casts
/// a transposed operand to the operator's *input* type first, so the
/// entries only it has take another path to the output than they do
/// untransposed. A lone `0.5` under `GT_UINT64` (`u64` in, `bool` out) is
/// `true` as a `bool`, but `0` as a `u64` and then `false`.
///
/// The specification has one answer, `(bool) 0.5`. If this test fails
/// GraphBLAS has settled on one, and `PassesThrough` can take any
/// relabelled operator.
#[test]
fn a_union_under_a_comparison_depends_on_orientation_at_this_pin() {
    setup();
    let b = Matrix::<bool>::new(2, 2).unwrap();
    let mut landed = Vec::new();
    for orientation in [crate::Orientation::ByRow, crate::Orientation::ByCol] {
        let mut a = matrix(2, 2, &[(0, 1, 0.5f64)]);
        a.set_orientation(orientation).unwrap();
        a.wait(crate::WaitMode::Materialize).unwrap();
        let c = Matrix::<u64>::new(2, 2).unwrap();
        // SAFETY: live handles of built-in types, which the C API accepts
        // in this combination; no mask, accumulator or descriptor.
        let info = unsafe {
            sys::GrB_Matrix_eWiseAdd_BinaryOp(
                c.as_raw(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                GT_UINT64.as_raw(),
                a.as_raw(),
                b.as_raw(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(info, GrB_Info::GrB_SUCCESS);
        landed.push((a.orientation() == c.orientation(), entries(&c)));
    }
    assert_eq!(
        landed,
        vec![(true, vec![(0, 1, 1)]), (false, vec![(0, 1, 0)])],
        "stored like the output the lone entry is (bool) 0.5; stored the \
         other way it went through u64"
    );
}

/// A typecast predicate compares each value cast to the operator's type.
/// GraphBLAS before `7e74fe5f78` (10.5.0) read the scalar's bytes as the
/// matrix's type in its factory kernels and kept nothing here: `1`, `5` and
/// `9` are all below the bits of `4.5` read as a `u64`.
#[test]
fn select_with_a_typecast_predicate_compares_in_the_operators_type() {
    setup();
    let a = matrix(2, 2, &[(0, 0, 1u64), (0, 1, 5), (1, 1, 9)]);
    let mut c = Matrix::<u64>::new(2, 2).unwrap();
    matrix::select(
        &mut c,
        Mask::NONE,
        None,
        &VALUEGT_FP64.typecast(),
        &a,
        4.5,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&c), vec![(0, 1, 5), (1, 1, 9)]);

    let u = vector(3, &[(0, 1u64), (2, 9)]);
    let mut w = Vector::<u64>::new(3).unwrap();
    vector::select(
        &mut w,
        Mask::NONE,
        None,
        &VALUEGT_FP64.typecast(),
        &u,
        4.5,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(vector_entries(&w), vec![(2, 9)]);
}

#[test]
fn a_user_defined_domain_stays_itself_under_typecast() {
    setup();
    // Only the output is cast: `Edge` reads as `Edge`, the f64 weight
    // lands in an i64 matrix.
    let a = matrix(
        1,
        2,
        &[
            (0, 0, Edge { weight: 1.5, id: 1 }),
            (0, 1, Edge { weight: 3.0, id: 2 }),
        ],
    );
    let mut weights = Matrix::<i64>::new(1, 2).unwrap();
    matrix::apply(
        &mut weights,
        Mask::NONE,
        None,
        &EDGE_WEIGHT.typecast::<Edge, i64>(),
        &a,
        Desc::NONE,
    )
    .unwrap();
    assert_eq!(entries(&weights), vec![(0, 0, 1), (0, 1, 3)]);

    // The identity relabelling is allowed and changes nothing.
    let mut heaviest = Edge { weight: 0.0, id: 0 };
    matrix::reduce(&mut heaviest, None, &EDGE_MAX.typecast(), &a, Desc::NONE).unwrap();
    assert_eq!(heaviest.id, 2);
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

#[test]
fn freeze_does_no_work_and_the_first_read_finishes_it() {
    setup();
    let m = matrix::<u64>(4, 4, &[(0, 1, 7), (2, 3, 9)]);
    assert!(m.is_pending(), "set_element left pending tuples");

    let frozen = m.freeze();
    assert!(!frozen.is_materialized());

    let m = frozen.read().unwrap();
    assert!(frozen.is_materialized());
    assert!(!m.is_pending());
    assert_eq!(entries(m), [(0, 1, 7), (2, 3, 9)]);
}

#[test]
fn a_frozen_matrix_is_an_input_and_a_mask_on_several_threads() {
    const READERS: usize = 8;
    setup();
    // A path 0 -> 1 -> 2 -> 3, read while its inserts are still pending.
    let edges = matrix::<bool>(4, 4, &[(0, 1, true), (1, 2, true), (2, 3, true)]).freeze();
    let seen = vector::<bool>(4, &[(1, true)]).freeze();

    let start = Barrier::new(READERS);
    thread::scope(|scope| {
        for _ in 0..READERS {
            let edges = edges.clone();
            let (seen, start) = (&seen, &start);
            scope.spawn(move || {
                start.wait();
                let edges = edges.read().unwrap();
                let seen = seen.read().unwrap();

                let mut two_hops = Matrix::<bool>::new(4, 4).unwrap();
                mxm(
                    &mut two_hops,
                    Mask::structural_complement(edges),
                    None,
                    &ANY_PAIR_BOOL,
                    edges,
                    edges,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(entries(&two_hops), [(0, 2, true), (1, 3, true)]);

                let mut next = Vector::<bool>::new(4).unwrap();
                vxm(
                    &mut next,
                    Mask::structural_complement(seen),
                    None,
                    &ANY_PAIR_BOOL,
                    seen,
                    edges,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(vector_entries(&next), [(2, true)]);

                assert_eq!(edges.extract_element(1, 2).unwrap(), Some(true));
                assert_eq!(edges.nvals(), 3);
            });
        }
    });
}

#[test]
fn a_frozen_hypersparse_matrix_is_read_through_its_hyper_hash() {
    const READERS: usize = 8;
    // More non-empty rows than the 1024 at which GraphBLAS builds, and then
    // looks rows up through, the hyper-hash.
    const ROWS: GrB_Index = 5000;
    const SIDE: GrB_Index = 1 << 40;
    setup();
    // Row `i << 20` points at column `i`.
    let mut wide = Matrix::<bool>::new(SIDE, SIDE).unwrap();
    for i in 0..ROWS {
        wide.set_element(i << 20, i, true).unwrap();
    }
    let wide = wide.freeze();

    let start = Barrier::new(READERS);
    thread::scope(|scope| {
        for _ in 0..READERS {
            let (wide, start) = (&wide, &start);
            scope.spawn(move || {
                start.wait();
                let wide = wide.read().unwrap();
                assert_eq!(wide.sparsity(), Sparsity::Hypersparse);
                for i in (0..ROWS).step_by(97) {
                    assert_eq!(wide.extract_element(i << 20, i).unwrap(), Some(true));
                    assert_eq!(wide.extract_element((i << 20) + 1, i).unwrap(), None);
                }
                // Two hops need a row that is also a column: only row 0.
                let mut two_hops = Matrix::<bool>::new(SIDE, SIDE).unwrap();
                mxm(
                    &mut two_hops,
                    Mask::NONE,
                    None,
                    &ANY_PAIR_BOOL,
                    wide,
                    wide,
                    Desc::NONE,
                )
                .unwrap();
                assert_eq!(entries(&two_hops), [(0, 0, true)]);
            });
        }
    });
}

#[test]
fn thaw_gives_the_matrix_back_for_writing() {
    setup();
    let frozen = matrix::<u64>(2, 2, &[(0, 0, 1)]).freeze();
    let reader = frozen.clone();
    assert_eq!(reader.read().unwrap().nvals(), 1);

    // Another clone still reads: copy to write.
    let frozen = frozen.try_thaw().expect_err("a reader holds a clone");
    let mut copy = frozen.read().unwrap().dup().unwrap();
    copy.set_element(1, 1, 2).unwrap();
    assert_eq!(
        reader.read().unwrap().nvals(),
        1,
        "the frozen one is untouched"
    );
    assert_eq!(copy.nvals(), 2);

    drop(reader);
    let mut m = frozen.try_thaw().expect("the last clone");
    m.set_element(1, 0, 3).unwrap();
    assert_eq!(entries(&m), [(0, 0, 1), (1, 0, 3)]);
}

/// `Frozen` is the only shareable form: the bounds, checked by the compiler.
#[test]
fn only_the_frozen_form_is_sync() {
    fn shareable<S: Send + Sync + Clone>() {}
    shareable::<Frozen<Matrix<u64>>>();
    shareable::<Frozen<Vector<Edge>>>();
}

#[test]
fn try_from_raw_adopts_only_a_handle_of_its_own_element_type() {
    setup();
    let raw = matrix::<u64>(2, 2, &[(0, 1, 7)]).into_raw();
    // SAFETY: a live matrix nothing else owns.
    let raw = unsafe { Matrix::<f64>::try_from_raw(raw) }.expect_err("holds u64, not f64");
    // SAFETY: handed back untouched, so still live and unowned.
    let raw = unsafe { Matrix::<Edge>::try_from_raw(raw) }.expect_err("not the user type");
    // SAFETY: as above.
    let m = unsafe { Matrix::<u64>::try_from_raw(raw) }.expect("its own type");
    assert_eq!(entries(&m), [(0, 1, 7)]);

    let raw = vector::<Edge>(3, &[]).into_raw();
    // SAFETY: a live vector nothing else owns.
    let raw = unsafe { Vector::<bool>::try_from_raw(raw) }.expect_err("holds Edge");
    // SAFETY: handed back untouched.
    let v = unsafe { Vector::<Edge>::try_from_raw(raw) }.expect("its own type");
    assert_eq!(v.size(), 3);
}
