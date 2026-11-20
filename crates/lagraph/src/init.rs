//! Starting LAGraph, and GraphBLAS with it.

use std::sync::atomic::{AtomicBool, Ordering};

use graphblas::{Allocators, ErrorKind};

use crate::{Error, Result, error::call, sys};

/// Set by [`init`], never cleared.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialise GraphBLAS and LAGraph together, optionally through custom
/// allocators. Call it **in place of** [`graphblas::init`], once; both
/// libraries then stay up until the process exits.
///
/// `LAGr_Init` starts GraphBLAS itself and then sets LAGraph up, and when
/// the second step fails it finalizes GraphBLAS again. Starting both in one
/// call means that can only happen before any matrix exists: nothing is
/// left behind holding memory from a library that went down.
///
/// The same thread rule as for [`graphblas::init`] applies: call it on a
/// thread that outlives the use of the library. LAGraph sets the GraphBLAS
/// thread count to `omp_get_max_threads()`; call
/// [`graphblas::set_nthreads`] after this, not before.
///
/// # Errors
///
/// `GrB_INVALID_VALUE` when GraphBLAS is already initialised (by this or by
/// [`graphblas::init`]); whatever `LAGr_Init` returns, in which case
/// neither library is up.
pub fn init(allocators: Option<Allocators>) -> Result<()> {
    graphblas::__start_with(|| {
        let mode = sys::GrB_Mode::GrB_NONBLOCKING;
        call(|msg| {
            allocators.map_or_else(
                // SAFETY: a `msg` buffer of `LAGRAPH_MSG_LEN` bytes.
                || unsafe { sys::LAGraph_Init(msg) },
                |allocators| {
                    let (malloc, calloc, realloc, free) = allocators.functions();
                    // SAFETY: as above; the allocators are function pointers
                    // that `Allocators::new` requires to stay valid.
                    unsafe {
                        sys::LAGr_Init(
                            mode,
                            Some(malloc),
                            Some(calloc),
                            Some(realloc),
                            Some(free),
                            msg,
                        )
                    }
                },
            )
        })?;
        INITIALIZED.store(true, Ordering::SeqCst);
        Ok::<(), Error>(())
    })
}

/// Whether [`init`] has run.
#[must_use]
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// `Err` if [`init`] has not run.
pub fn ensure_initialized() -> Result<()> {
    if is_initialized() {
        Ok(())
    } else {
        Err(Error::graphblas(
            ErrorKind::Panic,
            "lagraph::init has not run",
        ))
    }
}
