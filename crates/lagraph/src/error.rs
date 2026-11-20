//! What a failed LAGraph call reports, and the one function every call goes
//! through to report it.

use std::{
    ffi::CStr,
    fmt,
    os::raw::{c_char, c_int},
};

use graphblas::ErrorKind;

use crate::sys;

/// The status of a failed call.
///
/// LAGraph returns one `int` that is either a GraphBLAS code or one of its
/// own (-1000 and below); an algorithm may also define codes of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// A GraphBLAS failure passed through, or raised by LAGraph with a
    /// GraphBLAS code (`GrB_NULL_POINTER`, `GrB_INVALID_INDEX`, ...).
    GraphBlas(ErrorKind),
    /// `LAGRAPH_INVALID_GRAPH`: not square, held by column, a cached
    /// property of the wrong shape or type.
    InvalidGraph,
    /// `LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED`: the algorithm needs an
    /// undirected graph, or a directed one known to be symmetric.
    SymmetricStructureRequired,
    /// `LAGRAPH_IO_ERROR`.
    IoError,
    /// `LAGRAPH_NOT_CACHED`: the algorithm needs a cached property; the
    /// message names it.
    NotCached,
    /// `LAGRAPH_NO_SELF_EDGES_ALLOWED`.
    NoSelfEdgesAllowed,
    /// `LAGRAPH_CONVERGENCE_FAILURE`: the iteration limit was reached.
    ConvergenceFailure,
    /// A code neither library names in its headers.
    Other(i32),
}

impl Status {
    pub(crate) const fn from_code(code: c_int) -> Self {
        match code {
            sys::LAGRAPH_INVALID_GRAPH => Self::InvalidGraph,
            sys::LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED => Self::SymmetricStructureRequired,
            sys::LAGRAPH_IO_ERROR => Self::IoError,
            sys::LAGRAPH_NOT_CACHED => Self::NotCached,
            sys::LAGRAPH_NO_SELF_EDGES_ALLOWED => Self::NoSelfEdgesAllowed,
            sys::LAGRAPH_CONVERGENCE_FAILURE => Self::ConvergenceFailure,
            other => match ErrorKind::from_code(other) {
                ErrorKind::Other(_) => Self::Other(other),
                kind => Self::GraphBlas(kind),
            },
        }
    }
}

/// A failed LAGraph call: its status, and the text LAGraph wrote into the
/// call's `msg` buffer.
///
/// The text is a description only for a condition LAGraph checks itself.
/// For a GraphBLAS failure it passes through, and for its bare assertions,
/// it is the C source file and line of the check, which names the machine
/// the library was built on: log it, do not show it to a client as is. It
/// can also be empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: Status,
    pub message: String,
}

/// `Result` of a LAGraph call.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn graphblas(
        kind: ErrorKind,
        message: &str,
    ) -> Self {
        Self {
            status: Status::GraphBlas(kind),
            message: message.to_owned(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "LAGraph: {:?}", self.status)
        } else {
            write!(f, "LAGraph: {:?}: {}", self.status, self.message)
        }
    }
}

impl std::error::Error for Error {}

/// A failure of a `graphblas` call made on LAGraph's behalf.
impl From<graphblas::Error> for Error {
    fn from(e: graphblas::Error) -> Self {
        Self {
            status: Status::GraphBlas(e.kind),
            message: e.message,
        }
    }
}

/// Run one LAGraph function: hand it a fresh `msg` buffer, and turn a
/// negative status into an [`Error`] carrying what it wrote there. A
/// non-negative status (success, or a warning such as
/// `LAGRAPH_CACHE_NOT_NEEDED`) is returned as is.
pub fn call(function: impl FnOnce(*mut c_char) -> c_int) -> Result<c_int> {
    let mut msg = [0 as c_char; sys::LAGRAPH_MSG_LEN as usize];
    let status = function(msg.as_mut_ptr());
    if status >= 0 {
        return Ok(status);
    }
    // LAGraph terminates what it writes; this makes the read below safe even
    // if a function did not.
    msg[msg.len() - 1] = 0;
    // SAFETY: a NUL-terminated buffer that lives until the end of this
    // function.
    let message = unsafe { CStr::from_ptr(msg.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(Error {
        status: Status::from_code(status),
        message,
    })
}
