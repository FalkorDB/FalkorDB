//! Library initialisation and the global settings.
//!
//! There is no `finalize`. GraphBLAS starts once per process and stays up
//! until it exits: a safe shutdown would have to wait for every handle on
//! every thread, and nothing gains from tearing it down early. User-defined
//! types and operators are created once and live as long.

use std::{
    os::raw::{c_int, c_void},
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    Error, Result,
    error::check,
    sys::{self, GrB_Info},
};

/// Memory functions handed to `GxB_init`, so GraphBLAS allocates through the
/// host's allocator (the engine passes Redis's, for memory accounting).
#[derive(Clone, Copy)]
pub struct Allocators {
    malloc: unsafe extern "C" fn(usize) -> *mut c_void,
    calloc: unsafe extern "C" fn(usize, usize) -> *mut c_void,
    realloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
}

impl Allocators {
    /// The four functions GraphBLAS allocates through.
    ///
    /// # Safety
    ///
    /// They must behave as the C functions of the same names do, on one
    /// heap: `free` and `realloc` accept whatever any of the others
    /// returned. GraphBLAS calls them from several threads at once and for
    /// as long as the library is up, so they must be thread-safe and stay
    /// valid until the process ends.
    #[must_use]
    pub const unsafe fn new(
        malloc: unsafe extern "C" fn(usize) -> *mut c_void,
        calloc: unsafe extern "C" fn(usize, usize) -> *mut c_void,
        realloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
        free: unsafe extern "C" fn(*mut c_void),
    ) -> Self {
        Self {
            malloc,
            calloc,
            realloc,
            free,
        }
    }
}

/// Set by [`init`], never cleared.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Held across [`init`], which GraphBLAS requires to run once.
static STARTING: Mutex<()> = Mutex::new(());

/// Initialise GraphBLAS in non-blocking mode, optionally through custom
/// allocators. Must run once, before any other call in this crate; every
/// built-in and user-defined type checks that it did.
///
/// The library then stays up until the process exits.
///
/// Call it on a thread that lives as long as the library is used, such as
/// the process's main thread. OpenMP binds its runtime to the thread that
/// first uses it, and with a statically linked libomp (the Linux builds) a
/// later `GrB_Global_get` or `set` from another thread crashes once that
/// first thread has exited.
///
/// # Errors
///
/// `GrB_INVALID_VALUE` when GraphBLAS is already initialised.
pub fn init(allocators: Option<Allocators>) -> Result<()> {
    let _starting = STARTING.lock().unwrap_or_else(PoisonError::into_inner);
    if is_initialized() {
        return Err(Error::from_info(
            GrB_Info::GrB_INVALID_VALUE,
            "graphblas::init has already run".to_owned(),
        ));
    }
    let mode = sys::GrB_Mode::GrB_NONBLOCKING as c_int;
    let info = allocators.map_or_else(
        || {
            // SAFETY: a plain call.
            unsafe { sys::GrB_init(mode) }
        },
        |a| {
            // SAFETY: a plain call; the allocators are function pointers,
            // valid for the life of the process.
            unsafe {
                sys::GxB_init(
                    mode,
                    Some(a.malloc),
                    Some(a.calloc),
                    Some(a.realloc),
                    Some(a.free),
                )
            }
        },
    );
    check(info)?;
    INITIALIZED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Whether [`init`] has run.
#[must_use]
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// Panic with `what` in the message if [`init`] has not run.
pub fn assert_initialized(what: &str) {
    assert!(
        is_initialized(),
        "GraphBLAS `{what}` used before `graphblas::init`"
    );
}

/// `Err` if [`init`] has not run, with `GrB_PANIC`, the code GraphBLAS itself
/// returns for a call before `GrB_init`.
pub fn ensure_initialized() -> Result<()> {
    if is_initialized() {
        Ok(())
    } else {
        Err(Error::from_info(
            GrB_Info::GrB_PANIC,
            "graphblas::init has not run".to_owned(),
        ))
    }
}

fn set_global(
    value: i32,
    field: c_int,
) -> Result<()> {
    ensure_initialized()?;
    // SAFETY: `GrB_GLOBAL` is the library's global object; the field is one
    // of its integer settings.
    check(unsafe { sys::GrB_Global_set_INT32(sys::GrB_GLOBAL, value, field) })
}

/// The number of threads GraphBLAS may use: `GxB_NTHREADS`, and OpenMP's own
/// count for the calling thread.
///
/// GraphBLAS gates its parallel loops on the first; the second is what
/// OpenMP itself starts, and it is the one a `fork` child must lower to 1,
/// since the thread team it inherited is not there.
///
/// # Errors
///
/// Not initialised.
#[allow(clippy::cast_possible_wrap, reason = "`GxB_NTHREADS` is 7086")]
pub fn set_nthreads(n: i32) -> Result<()> {
    set_global(n, sys::GxB_NTHREADS as c_int)?;
    // SAFETY: a plain call into the OpenMP runtime GraphBLAS is linked with.
    unsafe { omp_set_num_threads(n) };
    Ok(())
}

unsafe extern "C" {
    /// From the OpenMP runtime that `graphblas-sys` links.
    fn omp_set_num_threads(num_threads: c_int);
}

/// `GxB_BURBLE`: print each operation's internal method choice to stdout.
///
/// # Errors
///
/// Not initialised.
pub fn set_burble(on: bool) -> Result<()> {
    set_global(i32::from(on), sys::GxB_Option_Field::GxB_BURBLE as c_int)
}

/// `GxB_JIT_C_CONTROL` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitControl {
    /// No JIT; PreJIT kernels are not used either.
    Off,
    /// Run no JIT kernel, but keep the loaded ones for later.
    Pause,
    /// Run loaded kernels, load nothing new.
    Run,
    /// Load compiled kernels, compile nothing.
    Load,
    /// Compile and load.
    On,
}

impl JitControl {
    const fn raw(self) -> i32 {
        use sys::GxB_JIT_Control as C;
        match self {
            Self::Off => C::GxB_JIT_OFF as i32,
            Self::Pause => C::GxB_JIT_PAUSE as i32,
            Self::Run => C::GxB_JIT_RUN as i32,
            Self::Load => C::GxB_JIT_LOAD as i32,
            Self::On => C::GxB_JIT_ON as i32,
        }
    }
}

/// `GxB_JIT_C_CONTROL`: how GraphBLAS uses its JIT and PreJIT kernels.
///
/// # Errors
///
/// Not initialised.
pub fn set_jit(control: JitControl) -> Result<()> {
    set_global(
        control.raw(),
        sys::GxB_Option_Field::GxB_JIT_C_CONTROL as c_int,
    )
}
