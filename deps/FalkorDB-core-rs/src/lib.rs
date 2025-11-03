/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

use std::alloc::{GlobalAlloc, Layout};
use std::os::raw::c_void;

mod binding;
mod undo_log;

extern "C" {
    static RedisModule_Alloc: Option<extern "C" fn(usize) -> *mut c_void>;
    static RedisModule_Free: Option<extern "C" fn(*mut c_void)>;
}

pub struct FalkorDBAlloc;

unsafe impl GlobalAlloc for FalkorDBAlloc {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        /*
         * To make sure the memory allocation by Redis is aligned to the according to the layout,
         * we need to align the size of the allocation to the layout.
         *
         * "Memory is conceptually broken into equal-sized chunks,
         * where the chunk size is a power of two that is greater than the page size.
         * Chunks are always aligned to multiples of the chunk size.
         * This alignment makes it possible to find metadata for user objects very quickly."
         *
         * From: https://linux.die.net/man/3/jemalloc
         */
        let size = (layout.size() + layout.align() - 1) & (!(layout.align() - 1));

        match RedisModule_Alloc {
            Some(alloc) => alloc(size).cast(),
            None => panic!("alloc function not found"),
        }
    }

    unsafe fn dealloc(
        &self,
        ptr: *mut u8,
        _layout: Layout,
    ) {
        match RedisModule_Free {
            Some(f) => f(ptr.cast()),
            None => panic!("free function not found"),
        };
    }
}

#[cfg(feature = "falkordb_allocator")]
#[global_allocator]
pub static ALLOC: FalkorDBAlloc = FalkorDBAlloc;
