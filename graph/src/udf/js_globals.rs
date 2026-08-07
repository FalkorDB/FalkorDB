//! # JS Global Object Setup
//!
//! This module configures the global JavaScript objects and functions that are
//! available to UDF scripts. There are two modes of operation, each with its
//! own setup function:
//!
//! ## Validation Mode ([`setup_validate_globals`])
//!
//! Used by [`js_context::validate_script`](super::js_context::validate_script)
//! to check user code in a throwaway context. In this mode:
//! - `falkor.register(name, func)` -- records the function name (for metadata)
//!   but does not persist anything.
//! - `falkor.log(...)` -- no-op (output is suppressed during validation).
//!
//! ## Runtime Mode ([`setup_runtime_globals`])
//!
//! Used when building the real per-thread execution context. In this mode:
//! - `falkor.register(name, func)` -- stores the function under a qualified
//!   key (`<library>.<name>`) in a JS-side registry object, so it can later be
//!   retrieved as a `Persistent<Function>`.
//! - `falkor.log(...)` -- prints the value to stderr via `eprintln!`.
//! - `graph.traverse(nodes, config?)` -- exposes the multi-source BFS
//!   traversal implemented in [`js_classes`](super::js_classes).
//!
//! ## JS Global Layout (Runtime)
//!
//! ```text
//! globalThis
//!   |-- falkor
//!   |     |-- register(name, func)   // stores to __falkor_registered_funcs
//!   |     '-- log(value)             // prints to stderr
//!   |-- graph
//!   |     '-- traverse(nodes, config?)
//!   |-- __falkor_registered_funcs    // { "lib.fn": Function, ... }
//!   '-- __falkor_current_lib         // set during library loading
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rquickjs::{Ctx, Function, Object, Persistent};

/// Set up globals for validation mode: falkor.register() just collects names.
pub fn setup_validate_globals(
    ctx: &Ctx<'_>,
    _names: Rc<RefCell<Vec<String>>>,
) -> Result<(), String> {
    let globals = ctx.globals();

    let falkor =
        Object::new(ctx.clone()).map_err(|e| format!("Failed to create falkor object: {e}"))?;

    // Create a JS-side registry array to collect names
    ctx.eval::<(), _>(
        "globalThis.__falkor_registered_names = [];\n\
         globalThis.__falkor_registered_funcs = {};\n",
    )
    .map_err(|e| format!("Failed to init registry: {e}"))?;

    // falkor.register stores name+func in global JS objects (avoids Rust lifetime issues)
    ctx.eval::<(), _>(
        "globalThis.__falkor_register = function(name, func) {\n\
             if (typeof func !== 'function') {\n\
                 throw new Error(\"Failed to register UDF library: second argument must be a function\");\n\
             }\n\
             if (globalThis.__falkor_registered_names.indexOf(name) >= 0) {\n\
                 throw new Error(\"Failed to register UDF library: function '\" + name + \"' already registered\");\n\
             }\n\
             globalThis.__falkor_registered_names.push(name);\n\
             globalThis.__falkor_registered_funcs[name] = func;\n\
         };",
    )
    .map_err(|e| format!("Failed to create register function: {e}"))?;

    let register_fn: Function = ctx
        .eval("globalThis.__falkor_register")
        .map_err(|e| format!("Failed to get register function: {e}"))?;
    falkor
        .set("register", register_fn)
        .map_err(|e| format!("Failed to set falkor.register: {e}"))?;

    // falkor.log - no-op in validation mode
    let log_fn = Function::new(ctx.clone(), |(): ()| -> () {})
        .map_err(|e| format!("Failed to create log function: {e}"))?;
    falkor
        .set("log", log_fn)
        .map_err(|e| format!("Failed to set falkor.log: {e}"))?;

    globals
        .set("falkor", falkor)
        .map_err(|e| format!("Failed to set global falkor: {e}"))?;

    Ok(())
}

/// After running validation scripts, collect the registered names from JS globals.
pub fn collect_validate_names(ctx: &Ctx<'_>) -> Result<Vec<String>, String> {
    let names_arr: rquickjs::Array = ctx
        .eval("globalThis.__falkor_registered_names")
        .map_err(|e| format!("Failed to get registered names: {e}"))?;
    let mut names = Vec::new();
    for i in 0..names_arr.len() {
        let name: String = names_arr.get(i).map_err(|e| format!("{e}"))?;
        names.push(name);
    }
    Ok(names)
}

/// Set up globals for runtime mode: falkor.register() stores to global JS object.
pub fn setup_runtime_globals(ctx: &Ctx<'_>) -> Result<(), String> {
    let globals = ctx.globals();

    let falkor =
        Object::new(ctx.clone()).map_err(|e| format!("Failed to create falkor object: {e}"))?;

    // Create a JS-side registry object keyed by qualified name (lib.func)
    ctx.eval::<(), _>(
        "globalThis.__falkor_registered_funcs = {};\n\
         globalThis.__falkor_current_lib = '';\n\
         globalThis.__falkor_register = function(name, func) {\n\
             if (typeof func !== 'function') {\n\
                 throw new Error(\"Failed to register UDF library: second argument must be a function\");\n\
             }\n\
             var qname = globalThis.__falkor_current_lib ? globalThis.__falkor_current_lib + '.' + name : name;\n\
             globalThis.__falkor_registered_funcs[qname] = func;\n\
         };",
    )
    .map_err(|e| format!("Failed to init runtime registry: {e}"))?;

    let register_fn: Function = ctx
        .eval("globalThis.__falkor_register")
        .map_err(|e| format!("Failed to get register function: {e}"))?;
    falkor
        .set("register", register_fn)
        .map_err(|e| format!("Failed to set falkor.register: {e}"))?;

    // falkor.log - prints to stderr
    let log_fn = Function::new(
        ctx.clone(),
        |val: rquickjs::Value<'_>| -> Result<(), rquickjs::Error> {
            let s = js_value_to_log_string(&val);
            eprintln!("{s}");
            Ok(())
        },
    )
    .map_err(|e| format!("Failed to create log function: {e}"))?;

    falkor
        .set("log", log_fn)
        .map_err(|e| format!("Failed to set falkor.log: {e}"))?;

    globals
        .set("falkor", falkor)
        .map_err(|e| format!("Failed to set global falkor: {e}"))?;

    // Set up the graph global object (traverse functionality)
    let graph_obj =
        Object::new(ctx.clone()).map_err(|e| format!("Failed to create graph object: {e}"))?;

    let traverse_fn = Function::new(ctx.clone(), crate::udf::js_classes::js_traverse)
        .map_err(|e| format!("Failed to create traverse function: {e}"))?;
    graph_obj
        .set("traverse", traverse_fn)
        .map_err(|e| format!("Failed to set graph.traverse: {e}"))?;

    let get_node_by_id_fn = Function::new(ctx.clone(), crate::udf::js_classes::js_get_node_by_id)
        .map_err(|e| format!("Failed to create getNodeById function: {e}"))?;
    graph_obj
        .set("getNodeById", get_node_by_id_fn)
        .map_err(|e| format!("Failed to set graph.getNodeById: {e}"))?;

    let iterate_nodes_fn = Function::new(ctx.clone(), crate::udf::js_classes::js_iterate_nodes)
        .map_err(|e| format!("Failed to create iterateNodes function: {e}"))?;
    graph_obj
        .set("iterateNodes", iterate_nodes_fn)
        .map_err(|e| format!("Failed to set graph.iterateNodes: {e}"))?;

    let iterate_edges_fn = Function::new(ctx.clone(), crate::udf::js_classes::js_iterate_edges)
        .map_err(|e| format!("Failed to create iterateEdges function: {e}"))?;
    graph_obj
        .set("iterateEdges", iterate_edges_fn)
        .map_err(|e| format!("Failed to set graph.iterateEdges: {e}"))?;

    globals
        .set("graph", graph_obj)
        .map_err(|e| format!("Failed to set global graph: {e}"))?;

    Ok(())
}

/// After running runtime scripts, collect registered function refs as Persistent values.
pub fn collect_runtime_funcs(
    ctx: &Ctx<'_>
) -> Result<HashMap<String, Persistent<Function<'static>>>, String> {
    let funcs_obj: Object = ctx
        .eval("globalThis.__falkor_registered_funcs")
        .map_err(|e| format!("Failed to get registered funcs: {e}"))?;

    let mut result = HashMap::new();
    let keys: Vec<String> = funcs_obj.keys::<String>().filter_map(Result::ok).collect();

    for key in keys {
        let val: rquickjs::Value = funcs_obj
            .get(&key)
            .map_err(|e| format!("Failed to get function '{key}': {e}"))?;
        if !val.is_function() {
            return Err(format!(
                "Expected a function for '{key}', got non-function value"
            ));
        }
        let func: Function = funcs_obj
            .get(&key)
            .map_err(|e| format!("Failed to get function '{key}': {e}"))?;
        let persistent = Persistent::save(ctx, func);
        result.insert(key, persistent);
    }

    Ok(result)
}

fn js_value_to_log_string(val: &rquickjs::Value<'_>) -> String {
    if val.is_null() {
        return "null".into();
    }
    if val.is_undefined() {
        return "undefined".into();
    }
    if let Some(b) = val.as_bool() {
        return b.to_string();
    }
    if let Some(i) = val.as_int() {
        return i.to_string();
    }
    if let Some(f) = val.as_float() {
        return f.to_string();
    }
    if let Some(s) = val.as_string() {
        return s.to_string().unwrap_or_default();
    }
    // For objects/arrays, try JSON.stringify
    if let Ok(json_str) = val.ctx().json_stringify(val.clone())
        && let Some(s) = json_str
    {
        return s.to_string().unwrap_or_default();
    }
    "[object]".into()
}
