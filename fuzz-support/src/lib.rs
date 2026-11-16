//! Setup shared by the fuzz harnesses (`runtime-fuzz-target`, `fuzz/`). A crate of its
//! own, so the engine (`graph`) carries no fuzz-only code.
//!
//! The harnesses run the engine in-process with no Redis. RediSearch calls the
//! `RedisModule_*` function pointers that Redis fills in when it loads a module,
//! so without Redis they are null and the first call (e.g. `rm_calloc` in
//! `CREATE INDEX`) jumps to address 0. [`init_index_support`] fills in the ones the
//! library-mode paths use and initialises RediSearch as `module_init` does.

// Dependency version duplicates are from transitive dependencies.
#![allow(clippy::multiple_crate_versions)]

use std::ffi::{c_char, c_int};
use std::ptr::null_mut;
use std::slice::from_raw_parts;
use std::sync::atomic::{AtomicUsize, Ordering};

use graph::index::redisearch::{
    REDISEARCH_INIT_LIBRARY, RediSearch_Init,
    redis::{
        REDISMODULE_OK, RedisModule_Alloc, RedisModule_Calloc, RedisModule_CreateString,
        RedisModule_CreateStringFromString, RedisModule_Free, RedisModule_FreeString,
        RedisModule_FreeThreadSafeContext, RedisModule_GetDetachedThreadSafeContext,
        RedisModule_HoldString, RedisModule_Log, RedisModule_Realloc, RedisModule_RetainString,
        RedisModule_Strdup, RedisModule_StringPtrLen, RedisModule_SubscribeToServerEvent,
        RedisModule_TrimStringAllocation, RedisModuleCtx, RedisModuleEvent,
        RedisModuleEventCallback, RedisModuleString,
    },
};
use graph::threadpool::init_thread_pool;

/// A `RedisModuleString` is opaque to RediSearch; the fuzz stand-in owns its bytes and
/// is reference counted because RediSearch retains and holds strings.
struct FuzzString {
    refs: AtomicUsize,
    bytes: Box<[u8]>,
}

impl FuzzString {
    fn alloc(bytes: Box<[u8]>) -> *mut RedisModuleString {
        Box::into_raw(Box::new(Self {
            refs: AtomicUsize::new(1),
            bytes,
        }))
        .cast()
    }
}

const unsafe extern "C" fn mock_get_detached_ctx(_ctx: *mut RedisModuleCtx) -> *mut RedisModuleCtx {
    null_mut()
}

const unsafe extern "C" fn mock_free_ctx(_ctx: *mut RedisModuleCtx) {}

const unsafe extern "C" fn mock_log(
    _ctx: *mut RedisModuleCtx,
    _level: *const c_char,
    _fmt: *const c_char,
) {
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "REDISMODULE_OK is the bindgen constant 0"
)]
const unsafe extern "C" fn mock_subscribe(
    _ctx: *mut RedisModuleCtx,
    _event: RedisModuleEvent,
    _callback: RedisModuleEventCallback,
) -> c_int {
    REDISMODULE_OK as c_int
}

unsafe extern "C" fn mock_create_string(
    _ctx: *mut RedisModuleCtx,
    ptr: *const c_char,
    len: usize,
) -> *mut RedisModuleString {
    let bytes = if len == 0 {
        Box::default()
    } else {
        // SAFETY: the caller passes `len` readable bytes.
        unsafe { from_raw_parts(ptr.cast::<u8>(), len) }.into()
    };
    FuzzString::alloc(bytes)
}

unsafe extern "C" fn mock_create_string_from_string(
    _ctx: *mut RedisModuleCtx,
    s: *const RedisModuleString,
) -> *mut RedisModuleString {
    // SAFETY: `s` came from `mock_create_string`.
    FuzzString::alloc(unsafe { &*s.cast::<FuzzString>() }.bytes.clone())
}

unsafe extern "C" fn mock_retain_string(
    _ctx: *mut RedisModuleCtx,
    s: *mut RedisModuleString,
) {
    // SAFETY: `s` came from a `FuzzString` constructor.
    unsafe { &*s.cast::<FuzzString>() }
        .refs
        .fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn mock_hold_string(
    ctx: *mut RedisModuleCtx,
    s: *mut RedisModuleString,
) -> *mut RedisModuleString {
    // SAFETY: as `mock_retain_string`.
    unsafe { mock_retain_string(ctx, s) };
    s
}

const unsafe extern "C" fn mock_trim_string_allocation(_s: *mut RedisModuleString) {}

unsafe extern "C" fn mock_string_ptr_len(
    s: *const RedisModuleString,
    len: *mut usize,
) -> *const c_char {
    // SAFETY: `s` came from `mock_create_string`; `len`, when non-null, is writable.
    let bytes = &unsafe { &*s.cast::<FuzzString>() }.bytes;
    if !len.is_null() {
        unsafe { len.write(bytes.len()) };
    }
    bytes.as_ptr().cast()
}

unsafe extern "C" fn mock_free_string(
    _ctx: *mut RedisModuleCtx,
    s: *mut RedisModuleString,
) {
    if s.is_null() {
        return;
    }
    // SAFETY: `s` came from a `FuzzString` constructor; the last owner frees it.
    unsafe {
        if (*s.cast::<FuzzString>())
            .refs
            .fetch_sub(1, Ordering::AcqRel)
            == 1
        {
            drop(Box::from_raw(s.cast::<FuzzString>()));
        }
    }
}

/// Makes index DDL and index-backed queries work without a server.
///
/// Points RediSearch's allocator at libc, stubs the server callbacks it makes while
/// indexing, initialises it in library mode, and starts the one-worker global thread pool
/// that index population and index drops are spawned on (the module starts it in
/// `module_init`). Call once, before any query runs.
///
/// The native index's background builder (`online_build`, feature `index-falkordb`) is
/// not started: as in the unit tests, its `enqueue` then does nothing, so no query can
/// meet an index that is still being built by it.
///
/// # Panics
/// If RediSearch fails to initialise.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "REDISMODULE_OK and REDISEARCH_INIT_LIBRARY are the bindgen constants 0 and 1"
)]
pub fn init_index_support() {
    // SAFETY: called once at harness start-up, before any other thread uses
    // RediSearch, so nothing reads the function pointers while they are written.
    // `RedisModule_Log` is variadic but its stub ignores every argument, which is
    // sound under the C ABI of every target we fuzz on.
    unsafe {
        RedisModule_Alloc = Some(libc::malloc);
        RedisModule_Calloc = Some(libc::calloc);
        RedisModule_Realloc = Some(libc::realloc);
        RedisModule_Free = Some(libc::free);
        RedisModule_Strdup = Some(libc::strdup);
        RedisModule_GetDetachedThreadSafeContext = Some(mock_get_detached_ctx);
        RedisModule_FreeThreadSafeContext = Some(mock_free_ctx);
        RedisModule_Log = Some(std::mem::transmute::<
            unsafe extern "C" fn(*mut RedisModuleCtx, *const c_char, *const c_char),
            unsafe extern "C" fn(*mut RedisModuleCtx, *const c_char, *const c_char, ...),
        >(mock_log));
        RedisModule_SubscribeToServerEvent = Some(mock_subscribe);
        RedisModule_CreateString = Some(mock_create_string);
        RedisModule_CreateStringFromString = Some(mock_create_string_from_string);
        RedisModule_RetainString = Some(mock_retain_string);
        RedisModule_HoldString = Some(mock_hold_string);
        RedisModule_TrimStringAllocation = Some(mock_trim_string_allocation);
        RedisModule_StringPtrLen = Some(mock_string_ptr_len);
        RedisModule_FreeString = Some(mock_free_string);
        let result = RediSearch_Init(null_mut(), REDISEARCH_INIT_LIBRARY as c_int);
        assert!(
            result == REDISMODULE_OK as c_int,
            "failed to initialise the RediSearch library"
        );
    }
    // Err only if already initialised, which is what a second call wants.
    let _ = init_thread_pool(1);
}
