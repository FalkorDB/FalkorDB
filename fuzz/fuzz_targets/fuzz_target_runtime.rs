// #![feature(c_variadic)]
#![no_main]

use std::collections::HashMap;

// use libc::{atexit, calloc, free, malloc, realloc, strdup};

use graph::{
    graph::{
        graph::Plan,
        graphblas::{GrB_Mode, GrB_init},
        mvcc_graph::MvccGraph,
    },
    // redisearch::{
    //     REDISEARCH_INIT_LIBRARY, RediSearch_Init,
    //     redis::{
    //         REDISMODULE_OK, RedisModule_Alloc, RedisModule_Calloc, RedisModule_CreateString,
    //         RedisModule_CreateStringPrintf, RedisModule_Free, RedisModule_FreeString,
    //         RedisModule_GetThreadSafeContext, RedisModule_Log, RedisModule_Realloc,
    //         RedisModule_Strdup, RedisModule_StringPtrLen, RedisModule_SubscribeToServerEvent,
    //         RedisModuleBlockedClient, RedisModuleCtx, RedisModuleEvent, RedisModuleEventCallback,
    //         RedisModuleString,
    //     },
    // },
    runtime::{eval::evaluate_param, functions::init_functions, pool::Pool, runtime::Runtime},
};
use libfuzzer_sys::{Corpus, fuzz_target};

// const unsafe extern "C" fn mock_get_thread_safe_context(
//     _bc: *mut RedisModuleBlockedClient
// ) -> *mut RedisModuleCtx {
//     null_mut()
// }

// unsafe extern "C" fn mock_printf(
//     _ctx: *mut RedisModuleCtx,
//     _fmt: *const c_char,
//     ...
// ) -> *mut RedisModuleString {
//     null_mut()
// }

// unsafe extern "C" fn mock_log(
//     _ctx: *mut RedisModuleCtx,
//     _level: *const ::std::os::raw::c_char,
//     _fmt: *const ::std::os::raw::c_char,
//     ...
// ) {
// }

// unsafe extern "C" fn mock_subscribe(
//     _ctx: *mut RedisModuleCtx,
//     _event: RedisModuleEvent,
//     _callback: RedisModuleEventCallback,
// ) -> ::std::os::raw::c_int {
//     REDISMODULE_OK as c_int
// }

// const unsafe extern "C" fn mock_create_string(
//     _ctx: *mut RedisModuleCtx,
//     _ptr: *const ::std::os::raw::c_char,
//     _len: usize,
// ) -> *mut RedisModuleString {
//     null_mut()
// }

// const unsafe extern "C" fn mock_string_ptr_len(
//     _str: *const RedisModuleString,
//     len: *mut usize,
// ) -> *const ::std::os::raw::c_char {
//     len.write(0);
//     null_mut()
// }

// const unsafe extern "C" fn mock_free_string(
//     _ctx: *mut RedisModuleCtx,
//     _str: *mut RedisModuleString,
// ) {
// }

// unsafe extern "C" {
//     fn RediSearch_CleanupModule();
// }

// extern "C" fn exit() {
//     unsafe {
//         GrB_finalize();
//         RediSearch_CleanupModule();
//     }
// }

fuzz_target!(init: {
        unsafe {
            GrB_init(GrB_Mode::GrB_NONBLOCKING as _);
            // RedisModule_Alloc = Some(malloc);
            // RedisModule_Calloc = Some(calloc);
            // RedisModule_Realloc = Some(realloc);
            // RedisModule_Free = Some(free);
            // RedisModule_FreeString = Some(mock_free_string);
            // RedisModule_GetThreadSafeContext = Some(mock_get_thread_safe_context);
            // RedisModule_CreateStringPrintf = Some(mock_printf);
            // RedisModule_CreateString = Some(mock_create_string);
            // RedisModule_Log = Some(mock_log);
            // RedisModule_Strdup = Some(strdup);
            // RedisModule_SubscribeToServerEvent = Some(mock_subscribe);
            // RedisModule_StringPtrLen = Some(mock_string_ptr_len);
            // let result = RediSearch_Init(null_mut(), REDISEARCH_INIT_LIBRARY as c_int);
            // assert!(result == REDISMODULE_OK as c_int, "Failed to initialize RediSearch library");
            // atexit(exit);
        }
        init_functions().expect("Failed to init functions");
    },|data: &[u8]| -> Corpus {
        let g = MvccGraph::new(1024, 1024, 25, "fuzz_test");
    std::str::from_utf8(data).map_or(Corpus::Reject, |query| {
        let Ok(Plan {
            plan, parameters, ..
        }) = g.read().borrow().get_plan(query)
        else {
            return Corpus::Reject;
        };
        let Ok(parameters) = parameters
            .into_iter()
            .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
            .collect::<Result<HashMap<_, _>, String>>()
        else {
            return Corpus::Reject;
        };
        let pool = Pool::new();
        let runtime = Runtime::new(g.read(), parameters, true, plan, false, String::new(), &pool, -1, false, None);
        match runtime.query() {
            Ok(_) => Corpus::Keep,
            _ => Corpus::Reject,
        }
    })
});
