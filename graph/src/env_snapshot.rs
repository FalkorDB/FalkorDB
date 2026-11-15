//! Owned, process-lifetime copies of a C `environ` array.
//!
//! The module repairs libc's `environ` when a server loads it with
//! `RTLD_DEEPBIND` and copy-relocates `environ` (issue #3493); see
//! `sync_deepbind_environ` in the module crate. The copy lives here, apart
//! from the loader-specific lookup, so it can be unit-tested on any platform.

use std::ffi::{CStr, c_char};

/// Deep-copy a NULL-terminated `char **` environment and leak the copy.
///
/// Both the pointer array and every string are fresh allocations that are
/// never freed, so the result stays valid whatever later happens to `src`: a
/// `setenv` that `realloc`s the array, or a `putenv` caller freeing its buffer
/// after `unsetenv`. A NULL `src` yields an empty environment (a lone NULL).
///
/// # Safety
///
/// `src` must be NULL or point to a NULL-terminated array of pointers to
/// NUL-terminated strings, none of which is modified or freed during the call.
#[must_use]
pub unsafe fn leak_deep_copy(src: *const *mut c_char) -> *mut *mut c_char {
    let mut len = 0;
    // SAFETY: the caller guarantees `src` is NULL or NULL-terminated.
    while !src.is_null() && unsafe { !(*src.add(len)).is_null() } {
        len += 1;
    }
    let mut copy: Vec<*mut c_char> = Vec::with_capacity(len + 1);
    for i in 0..len {
        // SAFETY: `i < len`, so the entry is a non-NULL pointer to a
        // NUL-terminated string, per the caller's contract.
        let entry = unsafe { CStr::from_ptr(*src.add(i)) };
        copy.push(entry.to_owned().into_raw());
    }
    copy.push(std::ptr::null_mut());
    Box::leak(copy.into_boxed_slice()).as_mut_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Read a NULL-terminated `char **` back into owned strings.
    unsafe fn read(env: *mut *mut c_char) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0;
        // SAFETY: `env` came from `leak_deep_copy`, which NULL-terminates it.
        while unsafe { !(*env.add(i)).is_null() } {
            out.push(
                unsafe { CStr::from_ptr(*env.add(i)) }
                    .to_str()
                    .unwrap()
                    .to_owned(),
            );
            i += 1;
        }
        out
    }

    #[test]
    fn null_source_is_empty() {
        let copy = unsafe { leak_deep_copy(std::ptr::null()) };
        assert_eq!(unsafe { read(copy) }, [] as [String; 0]);
    }

    #[test]
    fn empty_source_is_empty() {
        let src = [std::ptr::null_mut::<c_char>()];
        let copy = unsafe { leak_deep_copy(src.as_ptr()) };
        assert_eq!(unsafe { read(copy) }, [] as [String; 0]);
    }

    #[test]
    fn copies_entries_in_order() {
        let a = CString::new("A=1").unwrap();
        let b = CString::new("KMP_BLOCKTIME=0").unwrap();
        let src = [
            a.as_ptr().cast_mut(),
            b.as_ptr().cast_mut(),
            std::ptr::null_mut(),
        ];
        let copy = unsafe { leak_deep_copy(src.as_ptr()) };
        assert_eq!(unsafe { read(copy) }, ["A=1", "KMP_BLOCKTIME=0"]);
    }

    /// The copy owns its strings and array: freeing or overwriting the
    /// source afterwards (as `unsetenv` + `free` of a `putenv` buffer, or a
    /// `setenv` realloc, would) must not change what the copy reads.
    #[test]
    fn survives_source_being_freed_and_overwritten() {
        let a = CString::new("PUTENV_OWNED=yes").unwrap().into_raw();
        let b = CString::new("B=2").unwrap();
        let mut src = vec![a, b.as_ptr().cast_mut(), std::ptr::null_mut()];
        let copy = unsafe { leak_deep_copy(src.as_ptr()) };
        assert_ne!(unsafe { *copy }, a, "strings must be copied, not borrowed");
        assert_ne!(copy, src.as_mut_ptr(), "the array must be copied");
        drop(unsafe { CString::from_raw(a) });
        src[0] = std::ptr::null_mut();
        drop(src);
        assert_eq!(unsafe { read(copy) }, ["PUTENV_OWNED=yes", "B=2"]);
    }
}
