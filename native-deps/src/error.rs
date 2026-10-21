//! A single string-carrying error type. This crate's failures are all "tell the
//! developer what went wrong and what to do about it", so there is nothing for
//! callers to match on.

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// `return Err(Error(format!(...)))`.
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err($crate::Error(::std::format!($($arg)*)))
    };
}

/// `Error(format!(...))`, for use with `ok_or_else` / `map_err`.
#[macro_export]
macro_rules! err {
    ($($arg:tt)*) => {
        $crate::Error(::std::format!($($arg)*))
    };
}
