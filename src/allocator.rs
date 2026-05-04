//! Thread-local memory allocation tracking for Redis module.
//!
//! This module provides a custom global allocator that wraps Redis's allocator
//! while tracking per-thread memory usage. This enables the `--track-memory`
//! flag in `GRAPH.QUERY` to report allocation statistics.
//!
//! # Design
//!
//! Uses thread-local storage (TLS) for counters to avoid synchronization overhead.
//! Each thread maintains its own allocation/deallocation counters that can be
//! independently enabled, disabled, and reset.
//!
//! Thread-local storage is ideal here because:
//! 1. No mutex contention in the hot allocation path
//! 2. Each query runs on a single thread, so per-thread tracking is sufficient
//! 3. Counters can be reset before each query for accurate per-query stats

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::thread_local;

/// Global allocator that delegates to Redis while tracking allocations per-thread.
///
/// This allocator wraps `redis_module::alloc::RedisAlloc` to ensure all memory
/// is allocated through Redis's memory management system (for proper memory
/// limits and tracking), while adding per-thread allocation counting.
///
/// # Usage
/// Registered as the global allocator via the `redis_module!` macro's
/// `allocator` parameter.
pub struct ThreadCountingAllocator;

thread_local! {
    /// Cumulative bytes allocated on this thread since last reset.
    static THREAD_ALLOCATED: Cell<usize> = const { Cell::new(0) };
    /// Cumulative bytes deallocated on this thread since last reset.
    static THREAD_DEALLOCATED: Cell<usize> = const { Cell::new(0) };
    /// Whether allocation tracking is currently enabled for this thread.
    /// Disabled during logging to prevent recursive tracking.
    static TRACKING_ENABLED: Cell<bool> = const { Cell::new(true) };
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        let ptr = unsafe {
            redis_module::alloc::RedisAlloc::alloc(&redis_module::alloc::RedisAlloc, layout)
        };
        if !ptr.is_null() {
            TRACKING_ENABLED.with(|enabled| {
                if enabled.get() {
                    THREAD_ALLOCATED.with(|c| c.set(c.get() + layout.size()));
                }
            });
        }
        ptr
    }

    unsafe fn dealloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
    ) {
        unsafe {
            redis_module::alloc::RedisAlloc::dealloc(&redis_module::alloc::RedisAlloc, ptr, layout);
        };
        TRACKING_ENABLED.with(|enabled| {
            if enabled.get() {
                THREAD_DEALLOCATED.with(|c| c.set(c.get() + layout.size()));
            }
        });
    }
}

// Helper functions for the current thread

/// Enables allocation tracking for the current thread.
pub fn enable_tracking() {
    TRACKING_ENABLED.with(|flag| flag.set(true));
}

/// Disables allocation tracking for the current thread.
///
/// Call this before operations that allocate memory but shouldn't be tracked
/// (e.g., logging the tracking results themselves).
pub fn disable_tracking() {
    TRACKING_ENABLED.with(|flag| flag.set(false));
}

/// Returns the current thread's allocation statistics.
///
/// # Returns
/// A tuple of (bytes_allocated, bytes_deallocated) since last reset.
/// Net memory usage = allocated - deallocated.
pub fn current_thread_usage() -> (usize, usize) {
    (
        THREAD_ALLOCATED.with(std::cell::Cell::get),
        THREAD_DEALLOCATED.with(std::cell::Cell::get),
    )
}

/// Returns the net memory usage (allocated - deallocated) for the current thread.
pub fn net_thread_usage() -> usize {
    let allocated = THREAD_ALLOCATED.with(std::cell::Cell::get);
    let deallocated = THREAD_DEALLOCATED.with(std::cell::Cell::get);
    allocated.saturating_sub(deallocated)
}

/// Resets allocation counters for the current thread to zero.
///
/// Call this before a query to measure per-query memory usage.
pub fn reset_counter() {
    THREAD_ALLOCATED.with(|c| c.set(0));
    THREAD_DEALLOCATED.with(|c| c.set(0));
}
