//! [`Frozen<M>`]: a matrix or vector that can be read from several threads.
//!
//! GraphBLAS lets user threads share an object only as an *input*, and only
//! once its pending work is finished (`GrB_wait` with `GrB_MATERIALIZE`); a
//! read of an unfinished object finishes it first, which is a write. The
//! engine used to keep that rule by hand: a `has_pending` flag set after
//! every mutating call, a lock, and a `wait` before every read. Here it is
//! two types.
//!
//! * [`Matrix<T>`](crate::Matrix) is exclusive: mutated through `&mut`,
//!   `Send`, not `Sync`, not `Clone`.
//! * [`Frozen<Matrix<T>>`](FrozenMatrix) is shared: `Send + Sync`, `Clone`
//!   is a reference count, and there is no way to mutate through it.
//!
//! [`Matrix::freeze`](crate::Matrix::freeze) moves from the first to the
//! second and does no work. The only way back to the object is
//! [`Frozen::read`], which finishes the pending work the first time any
//! thread calls it and is one atomic load afterwards. Nothing can add
//! pending work after that, so the wait happens at most once, and only for
//! an object somebody reads.

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use crate::{Matrix, Result, Vector, sealed::Sealed};

/// A [`Matrix`] or [`Vector`]: what [`Frozen`] can hold. Sealed.
///
/// # Safety
///
/// Once `materialize` has returned `Ok`, every method the type offers
/// through `&self` (its own, and its use as an operand or a mask) only reads
/// the object inside GraphBLAS, so that any number of threads may call them
/// at once. An implementation that adds a `&self` method must keep that
/// true.
pub unsafe trait Freezable: Sealed + Send {
    /// `GrB_wait (object, GrB_MATERIALIZE)`.
    ///
    /// # Safety
    ///
    /// No other thread is using the object.
    #[doc(hidden)]
    unsafe fn materialize(&self) -> Result<()>;
}

/// A matrix or vector that is read-only and shared.
///
/// Made by [`Matrix::freeze`](crate::Matrix::freeze) or
/// [`Vector::freeze`](crate::Vector::freeze). Cloning shares the same
/// GraphBLAS object, which is freed when the last clone is dropped.
///
/// ```
/// # use graphblas::*;
/// # init(None).unwrap();
/// # set_jit(JitControl::Off).unwrap();
/// let mut edges = Matrix::<bool>::new(4, 4)?;
/// edges.set_element(0, 1, true)?;
/// let edges = edges.freeze();
///
/// let two_hops: Vec<Result<u64>> = std::thread::scope(|scope| {
///     let readers: Vec<_> = (0..2)
///         .map(|_| {
///             scope.spawn(|| {
///                 let edges = edges.read()?; // the first caller finishes the pending insert
///                 let mut reach = Matrix::<bool>::new(4, 4)?;
///                 mxm(&mut reach, Mask::NONE, None, &ANY_PAIR_BOOL, edges, edges, Desc::NONE)?;
///                 Ok(reach.nvals())
///             })
///         })
///         .collect();
///     readers.into_iter().map(|reader| reader.join().unwrap()).collect()
/// });
/// for count in two_hops {
///     assert_eq!(count?, 0); // one edge, so no path of length two
/// }
/// # Ok::<(), Error>(())
/// ```
///
/// A frozen object cannot be written, on any thread:
///
/// ```compile_fail,E0596
/// # use graphblas::*;
/// let edges = Matrix::<bool>::new(4, 4).unwrap().freeze();
/// edges.read().unwrap().set_element(0, 1, true).unwrap();
/// ```
///
/// and an object that is not frozen cannot be shared:
///
/// ```compile_fail,E0277
/// # use graphblas::*;
/// let edges = Matrix::<bool>::new(4, 4).unwrap();
/// std::thread::scope(|scope| {
///     scope.spawn(|| edges.nvals());
/// });
/// ```
pub struct Frozen<M> {
    shared: Arc<Shared<M>>,
}

/// A frozen [`Matrix<T>`].
pub type FrozenMatrix<T> = Frozen<Matrix<T>>;
/// A frozen [`Vector<T>`].
pub type FrozenVector<T> = Frozen<Vector<T>>;

struct Shared<M> {
    object: M,
    /// The outcome of the one materializing wait, once it has run. Never
    /// unset: nothing can reach `object` mutably while it is in here. A
    /// failure is kept too, because GraphBLAS clears an object whose wait
    /// fails (`GB_wait` frees its content on any error), so there is
    /// nothing left to try again on.
    materialized: OnceLock<Result<()>>,
}

#[allow(
    clippy::non_send_fields_in_send_ty,
    reason = "the field is an `Arc` of an object that is `Send` and not `Sync`; `read` is what makes sharing it sound"
)]
// SAFETY: the object is `Send`, and the last clone may drop it on any
// thread.
unsafe impl<M: Freezable> Send for Frozen<M> {}

// SAFETY: `&Frozen<M>` hands out `&M`, and only from `read`, after
// `materialize` succeeded with every other reader blocked. From then on
// the object is never written: `&M` cannot mutate it and `try_thaw` needs
// the last clone by value. That is the state in which GraphBLAS allows
// concurrent use: after `GrB_MATERIALIZE` "different user threads may
// simultaneously call GraphBLAS operations that use the matrix C as an
// input parameter" (user guide, GrB_Matrix_wait), and `Freezable` promises
// that `&M` offers nothing else.
unsafe impl<M: Freezable> Sync for Frozen<M> {}

impl<M: Freezable> Frozen<M> {
    pub(crate) fn new(object: M) -> Self {
        Self {
            shared: Arc::new(Shared {
                object,
                materialized: OnceLock::new(),
            }),
        }
    }

    /// The object, with its pending work finished.
    ///
    /// The first call, on whichever thread makes it, runs
    /// `GrB_wait (object, GrB_MATERIALIZE)`; callers that arrive meanwhile
    /// wait for it. Every later call is one atomic load.
    ///
    /// # Errors
    ///
    /// Out of memory while assembling. That is final: GraphBLAS has cleared
    /// the object, so this call and every later one return the same error
    /// and the object is never shared.
    pub fn read(&self) -> Result<&M> {
        let shared = &*self.shared;
        let outcome = shared.materialized.get_or_init(|| {
            // SAFETY: `get_or_init` runs this once and blocks every other
            // caller of `read` until it returns, so no thread has been
            // given `&M` yet.
            unsafe { shared.object.materialize() }
        });
        match outcome {
            Ok(()) => Ok(&shared.object),
            Err(error) => Err(error.clone()),
        }
    }

    /// Whether the wait has run and succeeded. `false` right after `freeze`.
    #[must_use]
    pub fn is_materialized(&self) -> bool {
        matches!(self.shared.materialized.get(), Some(Ok(())))
    }

    /// The object back, mutable again, if this is the only clone left;
    /// otherwise `self` unchanged. To write while other clones still read,
    /// copy it: `frozen.read()?.dup()`.
    ///
    /// After a failed [`read`](Self::read) the object is one GraphBLAS has
    /// cleared and invalidated: thaw it only to drop it.
    ///
    /// # Errors
    ///
    /// `Err(self)` when another clone exists.
    pub fn try_thaw(self) -> std::result::Result<M, Self> {
        Arc::try_unwrap(self.shared)
            .map(|shared| shared.object)
            .map_err(|shared| Self { shared })
    }
}

impl<M> Clone for Frozen<M> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<M> fmt::Debug for Frozen<M> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Not the object itself: printing it would read it before the wait.
        f.debug_struct("Frozen")
            .field("shared", &Arc::as_ptr(&self.shared))
            .field(
                "materialized",
                &matches!(self.shared.materialized.get(), Some(Ok(()))),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! The once itself, on a stand-in object that counts its waits. The
    //! behaviour against GraphBLAS is in `crate::tests`.

    use std::{
        sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::Duration,
    };

    use super::{Freezable, Frozen};
    use crate::{Error, ErrorKind, Result, sealed::Sealed, sys::GrB_Info};

    #[derive(Debug, Default)]
    struct Counted {
        waits: AtomicUsize,
        /// How many of the first waits fail.
        failures: usize,
    }

    impl Sealed for Counted {}

    // SAFETY: nothing is read through `&Counted` but the atomic counter.
    unsafe impl Freezable for Counted {
        unsafe fn materialize(&self) -> Result<()> {
            let attempt = self.waits.fetch_add(1, Ordering::SeqCst);
            // Long enough for every other reader to pile up behind this one.
            thread::sleep(Duration::from_millis(20));
            if attempt < self.failures {
                return Err(Error::from_info(GrB_Info::GrB_OUT_OF_MEMORY, String::new()));
            }
            Ok(())
        }
    }

    #[test]
    fn readers_racing_for_the_first_read_wait_once() {
        const READERS: usize = 8;
        let frozen = Frozen::new(Counted::default());
        assert!(!frozen.is_materialized(), "freeze does no work");

        let start = Barrier::new(READERS);
        thread::scope(|scope| {
            for _ in 0..READERS {
                scope.spawn(|| {
                    start.wait();
                    let object = frozen.read().unwrap();
                    assert_eq!(object.waits.load(Ordering::SeqCst), 1);
                });
            }
        });

        assert!(frozen.is_materialized());
        assert_eq!(frozen.read().unwrap().waits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failed_wait_is_final_and_shares_nothing() {
        let frozen = Frozen::new(Counted {
            failures: 1,
            ..Counted::default()
        });

        let error = frozen.read().expect_err("the wait fails");
        assert_eq!(error.kind, ErrorKind::OutOfMemory);
        assert!(!frozen.is_materialized());

        // GraphBLAS has cleared the object by now: no second wait.
        assert_eq!(frozen.read().expect_err("still failed"), error);
        assert!(!frozen.is_materialized());
        let object = frozen.try_thaw().expect("the last clone");
        assert_eq!(object.waits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn thaw_needs_the_last_clone() {
        let frozen = Frozen::new(Counted::default());
        let other = frozen.clone();

        let frozen = frozen.try_thaw().expect_err("another clone exists");
        drop(other);
        let object = frozen.try_thaw().expect("the last clone");
        assert_eq!(object.waits.load(Ordering::SeqCst), 0, "never read");
    }
}
