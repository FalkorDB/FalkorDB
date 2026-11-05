//! Encode log text so the pinned redis-module logger cannot panic on NUL.

use redis_module::Context;
use redis_module::logging::{log_notice, log_warning};
use std::borrow::Cow;

/// RedisModule_Log takes a C string. The pinned `redis-module` crate converts
/// the message with `CString::new(...).unwrap()`, which panics on an interior
/// NUL. Graph keys can hold those bytes after a rename, and logging the key
/// then exits the process (#3651).
///
/// Resource identity still uses the original key bytes. This is only the
/// diagnostic form: each interior NUL becomes the two-character sequence `\0`.
#[must_use]
pub fn for_redis(message: &str) -> Cow<'_, str> {
    if message.contains('\0') {
        Cow::Owned(message.replace('\0', "\\0"))
    } else {
        Cow::Borrowed(message)
    }
}

pub fn notice(message: impl AsRef<str>) {
    log_notice(for_redis(message.as_ref()).as_ref());
}

pub fn warning(message: impl AsRef<str>) {
    log_warning(for_redis(message.as_ref()).as_ref());
}

pub fn context_notice(
    ctx: &Context,
    message: impl AsRef<str>,
) {
    ctx.log_notice(for_redis(message.as_ref()).as_ref());
}

pub fn context_warning(
    ctx: &Context,
    message: impl AsRef<str>,
) {
    ctx.log_warning(for_redis(message.as_ref()).as_ref());
}

#[cfg(test)]
mod tests {
    use super::for_redis;
    use std::ffi::CString;

    #[test]
    fn interior_nul_is_escaped_for_redis_log() {
        assert_eq!(for_redis("ok").as_ref(), "ok");
        assert_eq!(for_redis("a\0b").as_ref(), "a\\0b");
        assert_eq!(for_redis("\0").as_ref(), "\\0");
        assert_eq!(for_redis("a\0b\0c").as_ref(), "a\\0b\\0c");
    }

    #[test]
    fn interior_nul_log_message_is_a_valid_c_string() {
        let raw = "Created 1 virtual keys for graph a\0b";
        assert!(
            CString::new(raw).is_err(),
            "the pinned redis-module logger panics on this message"
        );
        let escaped = for_redis(raw);
        assert!(CString::new(escaped.as_ref()).is_ok());
        assert!(!escaped.as_bytes().contains(&0));
    }
}
