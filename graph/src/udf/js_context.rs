//! # JS Execution Context Management
//!
//! This module manages thread-local QuickJS runtimes and provides the bridge
//! between the Rust query evaluator and JavaScript UDF execution.
//!
//! ## Thread-Local Context Lifecycle
//!
//! Each thread maintains its own `(Runtime, Context, cached functions)` tuple
//! in a thread-local `JS_STATE`. The context is lazily built and automatically
//! rebuilt whenever the global [`UdfRepo`](super::repository::UdfRepo) version
//! changes (e.g., after a library is loaded or deleted).
//!
//! ```text
//!   call_udf_bridge("mylib.myfunc", runtime, args)
//!       |
//!       v
//!   ensure_context_current()
//!       |-- repo.version() != cached version?
//!       |       yes --> rebuild_context()
//!       |                  |-- create new QuickJS Runtime + Context
//!       |                  |-- setup_runtime_globals()  (js_globals)
//!       |                  |-- evaluate all library scripts
//!       |                  '-- cache Persistent<Function> refs
//!       v
//!   look up function by qualified name ("mylib.myfunc")
//!       |
//!       v
//!   convert Rust args --> JS values  (type_convert::value_to_js)
//!       |
//!       v
//!   call JS function with timeout interrupt handler
//!       |
//!       v
//!   convert JS result --> Rust Value (type_convert::js_to_value)
//! ```
//!
//! ## Script Validation
//!
//! [`validate_script`] runs user code in a disposable QuickJS context to
//! check for syntax/runtime errors and collect the list of function names
//! registered via `falkor.register()`, without affecting the main context.
//!
//! ## Configuration
//!
//! Three atomic statics control QuickJS resource limits:
//! - `JS_HEAP_SIZE`  -- maximum heap memory (default 256 MiB)
//! - `JS_STACK_SIZE` -- maximum native stack (default 1 MiB)
//! - `JS_TIMEOUT_MS` -- per-call execution timeout (0 = unlimited)
//!
//! These can be changed at runtime; the next UDF call will pick up the new
//! values (a version bump forces context rebuild with updated limits).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};

use rquickjs::{CatchResultExt, CaughtError, Context, Function, Persistent, Runtime as JsRuntime};

use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use crate::udf::get_udf_repo;
use crate::udf::js_classes::clear_current_graph;
use crate::udf::js_globals;
use crate::udf::type_convert;

/// Extract a human-readable error message from a CaughtError.
/// If `include_name` is true, prefix with error type (e.g., "SyntaxError: ...").
fn caught_error_message(
    err: &CaughtError<'_>,
    include_name: bool,
) -> String {
    match err {
        CaughtError::Error(e) => format!("{e}"),
        CaughtError::Exception(ex) => {
            let mut msg = ex.message().unwrap_or_default();
            // rquickjs ReferenceError message: "foo is not defined"
            // C QuickJS format: "'foo' is not defined"
            // Normalize to match the C format for compatibility.
            if let Some(stripped) = msg.strip_suffix(" is not defined")
                && !stripped.starts_with('\'')
            {
                msg = format!("'{stripped}' is not defined");
            }
            if include_name {
                // Try to get the error name (SyntaxError, TypeError, ReferenceError, etc.)
                let name = ex.as_object().get::<_, String>("name").ok();
                match name {
                    Some(n) if !n.is_empty() && n != "Error" => format!("{n}: {msg}"),
                    _ => msg,
                }
            } else {
                msg
            }
        }
        CaughtError::Value(val) => val.as_string().map_or_else(
            || format!("{val:?}"),
            |s| s.to_string().unwrap_or_else(|_| format!("{val:?}")),
        ),
    }
}

/// Atomic copies of JS config values, accessible without Redis GIL.
pub static JS_HEAP_SIZE: AtomicI64 = AtomicI64::new(256 * 1024 * 1024);
pub static JS_STACK_SIZE: AtomicI64 = AtomicI64::new(1024 * 1024);
/// Per-call JS execution timeout in milliseconds.
///
/// `0` preserves the historical "unlimited" contract that operators can opt
/// into, but in that case callers fall back to [`JS_TIMEOUT_ABSOLUTE_CAP_MS`]
/// so a runaway UDF cannot permanently consume a worker (or the Redis main
/// thread during `GRAPH.UDF LOAD` validation).
pub static JS_TIMEOUT_MS: AtomicI64 = AtomicI64::new(5_000);

/// Hard upper bound used when the configured timeout is `0` (unlimited) or
/// when validating user-submitted scripts on the Redis main thread.
pub const JS_TIMEOUT_ABSOLUTE_CAP_MS: u64 = 30_000;
/// Tighter cap for `validate_script`, which runs on the Redis main thread.
pub const JS_VALIDATE_CAP_MS: u64 = 10_000;

/// Resolve the per-call JS execution timeout, applying the absolute cap.
///
/// `JS_TIMEOUT_MS == 0` is the operator's "unlimited" opt-in; in that case
/// we still bound the deadline by `JS_TIMEOUT_ABSOLUTE_CAP_MS` so a runaway
/// script cannot hold a worker forever. Returns the effective timeout in
/// milliseconds; both the QuickJS interrupt handler in `call_udf_bridge`
/// and the BFS deadline in `js_classes` should derive from this value so
/// the cap cannot be bypassed.
#[must_use]
pub fn compute_effective_js_timeout_ms() -> u64 {
    let timeout_ms = JS_TIMEOUT_MS.load(Ordering::Relaxed);
    if timeout_ms > 0 {
        (timeout_ms as u64).min(JS_TIMEOUT_ABSOLUTE_CAP_MS)
    } else {
        JS_TIMEOUT_ABSOLUTE_CAP_MS
    }
}

struct ThreadJsState {
    runtime: JsRuntime,
    context: Context,
    /// Cached function references: "lib.func" -> persistent JS function
    functions: HashMap<String, Persistent<Function<'static>>>,
    /// Version of the UdfRepo when this context was last rebuilt.
    version: u64,
}

thread_local! {
    static JS_STATE: RefCell<Option<ThreadJsState>> = const { RefCell::new(None) };
}

/// Validate a JS script by running it in a temporary context.
/// Returns the list of function names registered via `falkor.register()`.
pub fn validate_script(code: &str) -> Result<Vec<String>, String> {
    let rt = JsRuntime::new().map_err(|e| format!("Failed to create JS runtime: {e}"))?;
    rt.set_memory_limit(JS_HEAP_SIZE.load(Ordering::Relaxed) as usize);
    rt.set_max_stack_size(JS_STACK_SIZE.load(Ordering::Relaxed) as usize);

    let timeout_ms = JS_TIMEOUT_MS.load(Ordering::Relaxed);
    // `validate_script` runs on the Redis main thread, so we must always
    // install an interrupt handler — a runaway user script would otherwise
    // freeze the entire server. Cap at `JS_VALIDATE_CAP_MS` regardless of
    // configuration.
    let effective_ms = if timeout_ms > 0 {
        (timeout_ms as u64).min(JS_VALIDATE_CAP_MS)
    } else {
        JS_VALIDATE_CAP_MS
    };
    let deadline = Instant::now() + Duration::from_millis(effective_ms);
    rt.set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)));

    let ctx = Context::full(&rt).map_err(|e| format!("Failed to create JS context: {e}"))?;

    ctx.with(|ctx| {
        let names = Rc::new(RefCell::new(Vec::new()));
        js_globals::setup_validate_globals(&ctx, names)?;

        ctx.eval::<(), _>(code)
            .catch(&ctx)
            .map_err(|e| caught_error_message(&e, true))?;

        js_globals::collect_validate_names(&ctx)
    })
}

/// Ensure the thread-local JS context is up-to-date with the global repository.
fn ensure_context_current() -> Result<(), String> {
    let repo = get_udf_repo();
    let current_version = repo.version();

    JS_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let needs_rebuild = (*state)
            .as_ref()
            .is_none_or(|s| s.version != current_version);

        if needs_rebuild {
            rebuild_context(&mut state, current_version)?;
        }
        Ok(())
    })
}

fn rebuild_context(
    state: &mut Option<ThreadJsState>,
    target_version: u64,
) -> Result<(), String> {
    // Drop old state in correct order: functions first, then context, then runtime.
    // Persistent references must be dropped while the runtime is still alive.
    if let Some(old) = state.take() {
        drop(old.functions);
        drop(old.context);
        drop(old.runtime);
    }

    let heap_size = JS_HEAP_SIZE.load(Ordering::Relaxed);
    let stack_size = JS_STACK_SIZE.load(Ordering::Relaxed);

    let rt = JsRuntime::new().map_err(|e| format!("Failed to create JS runtime: {e}"))?;
    rt.set_memory_limit(heap_size as usize);
    rt.set_max_stack_size(stack_size as usize);

    let ctx = Context::full(&rt).map_err(|e| format!("Failed to create JS context: {e}"))?;

    let functions = ctx.with(|ctx| {
        // Set up runtime globals (falkor.register stores to JS global object)
        js_globals::setup_runtime_globals(&ctx)?;

        // Evaluate all library scripts
        let repo = get_udf_repo();
        let libs = repo.get_all_libraries();
        for lib in &libs {
            // Set current library name so falkor.register() creates qualified keys
            ctx.eval::<(), _>(format!("globalThis.__falkor_current_lib = {:?};", lib.name))
                .map_err(|e| format!("Failed to set current lib: {e}"))?;

            ctx.eval::<(), _>(lib.code.as_str())
                .catch(&ctx)
                .map_err(|e| {
                    format!(
                        "Failed to load UDF library '{}': {}",
                        lib.name,
                        caught_error_message(&e, true)
                    )
                })?;
        }

        // Reset current lib name
        ctx.eval::<(), _>("globalThis.__falkor_current_lib = '';")
            .map_err(|e| format!("Failed to reset current lib: {e}"))?;

        // Collect function refs from JS global registry
        let raw_funcs = js_globals::collect_runtime_funcs(&ctx)?;

        // Map qualified names (lib.func) to their Persistent function references
        // Functions are now stored under qualified keys in JS registry
        let mut persistent_funcs = HashMap::new();
        for lib in &libs {
            for qname in &lib.function_names {
                if let Some(persistent) = raw_funcs.get(qname) {
                    persistent_funcs.insert(qname.to_lowercase(), persistent.clone());
                }
            }
        }

        Ok::<_, String>(persistent_funcs)
    })?;

    *state = Some(ThreadJsState {
        runtime: rt,
        context: ctx,
        functions,
        version: target_version,
    });

    Ok(())
}

/// Call a UDF by its qualified name (e.g., "mylib.myfunc").
/// This is called from the eval path when a UDF GraphFn is invoked.
pub fn call_udf_bridge(
    name: &str,
    rt: &Runtime,
    args: &[Value],
) -> Result<Value, String> {
    ensure_context_current()?;

    JS_STATE.with(|state| {
        let state = state.borrow();
        let state = state.as_ref().ok_or("JS context not initialized")?;

        let lower_name = name.to_lowercase();
        let persistent_fn = state
            .functions
            .get(&lower_name)
            .ok_or_else(|| format!("UDF function '{name}' not found in JS context"))?;

        // Set up timeout interrupt handler. We always install one: a configured
        // timeout of 0 (unlimited) falls back to the absolute cap so a runaway
        // UDF cannot hold a worker forever.
        let effective_ms = compute_effective_js_timeout_ms();
        let deadline = Instant::now() + Duration::from_millis(effective_ms);
        state
            .runtime
            .set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)));

        let result = state.context.with(|ctx| {
            let js_fn: Function = persistent_fn
                .clone()
                .restore(&ctx)
                .map_err(|e| format!("Failed to restore UDF function: {e}"))?;

            // Set graph reference for JS classes that need it
            crate::udf::js_classes::set_current_graph(rt.g.clone());

            // Convert arguments
            let js_args: Vec<rquickjs::Value> = args
                .iter()
                .map(|v| type_convert::value_to_js(&ctx, v, &rt.g))
                .collect::<Result<Vec<_>, _>>()?;

            // Call the function
            let result = js_fn
                .call::<(rquickjs::function::Rest<rquickjs::Value>,), rquickjs::Value>((
                    rquickjs::function::Rest(js_args),
                ))
                .catch(&ctx)
                .map_err(|e| {
                    let msg = caught_error_message(&e, false);
                    if msg.contains("interrupted") {
                        "UDF Exception: Query timed out".to_string()
                    } else if msg.contains("out of memory")
                        || msg.contains("InternalError: stack overflow")
                    {
                        "out of memory".to_string()
                    } else {
                        format!("UDF Exception: {msg}")
                    }
                })?;

            // Convert result back
            type_convert::js_to_value(result)
        });

        // Clear interrupt handler
        state
            .runtime
            .set_interrupt_handler(None::<Box<dyn FnMut() -> bool + Send>>);

        // Clear graph reference
        clear_current_graph();

        result
    })
}
