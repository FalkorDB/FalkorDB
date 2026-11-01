//! # Rust <-> JavaScript Type Conversion
//!
//! This module provides bidirectional conversion between FalkorDB's Rust
//! [`Value`](crate::runtime::value::Value) type and QuickJS JavaScript values.
//! It is used whenever arguments are passed into a UDF and when the UDF result
//! is returned back to the Cypher runtime.
//!
//! ## Conversion Table
//!
//! ```text
//! Rust Value               JS Value                   Notes
//! ----------               --------                   -----
//! Null                 <-> null / undefined
//! Bool(b)              <-> boolean
//! Int(i)               <-> number (f64)               i64 within +/-(2^53-1)
//! Int(i)               <-> BigInt                     i64 outside safe range
//! Float(f)             <-> number (f64)
//! String(s)            <-> string
//! List([...])          <-> Array [...]
//! Map({k: v})          <-> Object {k: v}              __falkor_* keys escaped
//! Node(id)             <-> Node object                see js_classes
//! Relationship(r,s,d)  <-> Edge object                see js_classes
//! Path([...])          <-> Path object                see js_classes
//! Point(lat, lon)      <-> {latitude, longitude}      __falkor_type = "point"
//! Datetime(ts)         <-> Date                       __falkor_temporal_type = "datetime"
//! Date(ts)             <-> Date                       __falkor_temporal_type = "date"
//! VecF32([...])        <-> Array (f64)                __falkor_type = "vecf32"
//! Time / Duration       -> (unsupported, returns Err)
//! ```
//!
//! ## Round-Trip Safety
//!
//! Graph entities (Node, Edge, Path, Point) carry hidden `__falkor_type` and
//! `__falkor_*_id` properties on their JS objects. These markers allow
//! [`js_to_value`] to reconstruct the correct Rust variant without ambiguity.
//! User-supplied map keys that start with `__falkor_` are escaped during
//! `value_to_js` and unescaped during `js_to_value` to prevent collisions.
//!
//! Integer round-tripping: JS numbers that equal their floor and fit within
//! the safe integer range are converted back to `Value::Int`, not `Value::Float`.

use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use rquickjs::function::This;
use rquickjs::{Array, Ctx, IntoJs, Object, Value as JsValue};

use crate::graph::graph::Graph;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{Point, Value};

/// The deepest a value may nest when it crosses into or out of a UDF: `[]` and
/// `{}` are depth 1, `[[]]` is depth 2.
///
/// Both conversions recurse once per level, and each level holds QuickJS
/// handles and an `rquickjs` frame, so a level here costs far more stack than
/// it does anywhere else in the engine. A UDF can build an arbitrarily deep
/// array or object in a loop (or one that contains itself), so without a bound
/// the conversion exhausts the stack of the query thread and aborts the
/// server. 1,024 is far beyond any nesting real data uses.
///
/// The bound alone does not fit every build in a 2 MiB worker stack: a level
/// costs about 0.9 KB on a release build but about 6 KB under ASAN, where
/// 1,024 levels overflow. So each level also runs through [`with_stack`],
/// which moves onto a fresh heap-allocated stack segment when the current one
/// runs low.
pub const MAX_UDF_VALUE_NESTING: usize = 1024;

/// Stack that must remain before a conversion level runs on the current
/// stack, with room for several sanitizer-sized levels.
const STACK_RED_ZONE: usize = 128 * 1024;

/// Size of each stack segment [`with_stack`] allocates when the current one
/// runs low.
const STACK_SEGMENT: usize = 1024 * 1024;

/// Runs one conversion level, first switching to a new stack segment if less
/// than [`STACK_RED_ZONE`] of the current one is left. The check is a
/// stack-pointer comparison; a segment is only allocated by a deep value.
fn with_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, f)
}

/// The error a conversion returns for a value nested past
/// [`MAX_UDF_VALUE_NESTING`].
#[must_use]
pub fn udf_nesting_error() -> String {
    format!("UDF value nesting exceeds the maximum depth of {MAX_UDF_VALUE_NESTING}")
}

/// Convert a FalkorDB Value to a JavaScript value.
///
/// `runtime`, when present, resolves relationship endpoints/type/attributes
/// through the deleted→pending→graph fallback so relationships created or
/// deleted earlier in the same query are handled correctly. Lazy JS graph
/// accessors that only have a committed-graph handle pass `None`.
pub fn value_to_js<'js>(
    ctx: &Ctx<'js>,
    value: &Value,
    graph: &Arc<AtomicRefCell<Graph>>,
    runtime: Option<&Runtime<'_>>,
) -> Result<JsValue<'js>, String> {
    value_to_js_at(ctx, value, graph, runtime, 0)
}

/// [`value_to_js`] for a value inside `depth` enclosing lists or maps.
///
/// The conversion recurses once per level, so a list or map past
/// [`MAX_UDF_VALUE_NESTING`] is an error rather than a stack overflow.
fn value_to_js_at<'js>(
    ctx: &Ctx<'js>,
    value: &Value,
    graph: &Arc<AtomicRefCell<Graph>>,
    runtime: Option<&Runtime<'_>>,
    depth: usize,
) -> Result<JsValue<'js>, String> {
    if matches!(value, Value::List(_) | Value::Map(_)) && depth >= MAX_UDF_VALUE_NESTING {
        return Err(udf_nesting_error());
    }
    with_stack(|| value_to_js_level(ctx, value, graph, runtime, depth))
}

/// The body of [`value_to_js_at`], run on a stack with room for it.
fn value_to_js_level<'js>(
    ctx: &Ctx<'js>,
    value: &Value,
    graph: &Arc<AtomicRefCell<Graph>>,
    runtime: Option<&Runtime<'_>>,
    depth: usize,
) -> Result<JsValue<'js>, String> {
    match value {
        Value::Null => Ok(JsValue::new_null(ctx.clone())),
        Value::Bool(b) => b
            .into_js(ctx)
            .map_err(|e| format!("JS conversion error: {e}")),
        Value::Int(i) => {
            // JS Number can safely represent integers up to ±(2^53 - 1).
            const MAX_SAFE_INTEGER: i64 = (1_i64 << 53) - 1;
            if *i >= -MAX_SAFE_INTEGER && *i <= MAX_SAFE_INTEGER {
                (*i as f64)
                    .into_js(ctx)
                    .map_err(|e| format!("JS conversion error: {e}"))
            } else {
                // Large integers -> BigInt
                ctx.eval::<JsValue, _>(format!("{i}n"))
                    .map_err(|e| format!("JS BigInt conversion error: {e}"))
            }
        }
        Value::Float(f) => f
            .into_js(ctx)
            .map_err(|e| format!("JS conversion error: {e}")),
        Value::String(s) => s
            .as_str()
            .into_js(ctx)
            .map_err(|e| format!("JS conversion error: {e}")),
        Value::List(items) => {
            let arr =
                Array::new(ctx.clone()).map_err(|e| format!("JS array creation error: {e}"))?;
            for (i, item) in items.iter().enumerate() {
                let js_val = value_to_js_at(ctx, item, graph, runtime, depth + 1)?;
                arr.set(i, js_val)
                    .map_err(|e| format!("JS array set error: {e}"))?;
            }
            Ok(arr.into_value())
        }
        Value::Map(map) => {
            let obj =
                Object::new(ctx.clone()).map_err(|e| format!("JS object creation error: {e}"))?;
            for (key, val) in map.iter() {
                let js_val = value_to_js_at(ctx, val, graph, runtime, depth + 1)?;
                // Escape keys that start with "__falkor_" to avoid collision
                // with internal metadata properties.
                let js_key = if key.starts_with("__falkor_") {
                    format!("__falkor_esc_{key}")
                } else {
                    key.to_string()
                };
                obj.set(js_key.as_str(), js_val)
                    .map_err(|e| format!("JS object set error: {e}"))?;
            }
            Ok(obj.into_value())
        }
        Value::Node(node_id) => {
            crate::udf::js_classes::create_js_node(ctx, (*node_id).into(), graph, runtime)
        }
        Value::Relationship(rel_id) => {
            let (src, dst) = runtime.map_or_else(
                || graph.borrow().get_relationship_endpoints(*rel_id),
                |rt| rt.get_relationship_endpoints(*rel_id),
            );
            crate::udf::js_classes::create_js_edge(
                ctx,
                (*rel_id).into(),
                src.into(),
                dst.into(),
                graph,
                runtime,
            )
        }
        Value::Path(path_values) => {
            crate::udf::js_classes::create_js_path(ctx, path_values, graph, runtime)
        }
        Value::Point(coords) => {
            let obj =
                Object::new(ctx.clone()).map_err(|e| format!("JS object creation error: {e}"))?;
            obj.set("__falkor_type", "point")
                .map_err(|e| format!("JS set error: {e}"))?;
            obj.set("latitude", coords.latitude as f64)
                .map_err(|e| format!("JS set error: {e}"))?;
            obj.set("longitude", coords.longitude as f64)
                .map_err(|e| format!("JS set error: {e}"))?;
            Ok(obj.into_value())
        }
        Value::Datetime(ts) => {
            let ms = *ts * 1000;
            let date: JsValue = ctx
                .eval(format!(
                    "(function() {{ var d = new Date({ms}); d.__falkor_temporal_type = 'datetime'; return d; }})()"
                ))
                .map_err(|e| format!("JS Date conversion error: {e}"))?;
            Ok(date)
        }
        Value::Date(ts) => {
            let ms = *ts * 1000;
            let date: JsValue = ctx
                .eval(format!(
                    "(function() {{ var d = new Date({ms}); d.__falkor_temporal_type = 'date'; return d; }})()"
                ))
                .map_err(|e| format!("JS Date conversion error: {e}"))?;
            Ok(date)
        }
        Value::VecF32(vec) => {
            let arr =
                Array::new(ctx.clone()).map_err(|e| format!("JS array creation error: {e}"))?;
            for (i, &v) in vec.iter().enumerate() {
                arr.set(i, v as f64)
                    .map_err(|e| format!("JS array set error: {e}"))?;
            }
            // Mark with non-enumerable __falkor_type so round-trip preserves VecF32
            arr.as_object()
                .set("__falkor_type", "vecf32")
                .map_err(|e| format!("JS set error: {e}"))?;
            Ok(arr.into_value())
        }
        Value::Time(_) | Value::Duration(_) => {
            Err("UDF Exception: unsupported type for JS conversion".to_string())
        }
    }
}

/// Convert a JavaScript value back to a FalkorDB Value.
pub fn js_to_value(val: JsValue<'_>) -> Result<Value, String> {
    js_to_value_at(val, 0)
}

/// [`js_to_value`] for a value inside `depth` enclosing arrays or objects.
///
/// A UDF can build an arbitrarily deep array or object in a loop (or one that
/// contains itself), and the conversion recurses once per level. So an array
/// or object past [`MAX_UDF_VALUE_NESTING`] is an error rather than a stack
/// overflow; that is also the bound every other way into the engine holds.
fn js_to_value_at(
    val: JsValue<'_>,
    depth: usize,
) -> Result<Value, String> {
    with_stack(|| js_to_value_level(val, depth))
}

/// The body of [`js_to_value_at`], run on a stack with room for it.
fn js_to_value_level(
    val: JsValue<'_>,
    depth: usize,
) -> Result<Value, String> {
    if val.is_null() || val.is_undefined() {
        return Ok(Value::Null);
    }

    if let Some(b) = val.as_bool() {
        return Ok(Value::Bool(b));
    }

    if val.is_int()
        && let Some(i) = val.as_int()
    {
        return Ok(Value::Int(i64::from(i)));
    }

    if let Some(f) = val.as_float() {
        // Check if it's an integer value
        // -0.0 stays a Float (as in C): as an Int it would lose its sign.
        if f == f.floor() && f.abs() < (1i64 << 53) as f64 && !(f == 0.0 && f.is_sign_negative()) {
            return Ok(Value::Int(f as i64));
        }
        return Ok(Value::Float(f));
    }

    // BigInt → Int
    if val.is_big_int() {
        let big_int = val
            .try_into_big_int()
            .map_err(|_| "Failed to convert BigInt".to_string())?;
        let i = big_int
            .to_i64()
            .map_err(|_| "BigInt out of i64 range".to_string())?;
        return Ok(Value::Int(i));
    }

    if let Some(s) = val.as_string() {
        let s = s.to_string().map_err(|e| format!("JS string error: {e}"))?;
        return Ok(Value::String(Arc::new(s)));
    }

    // Check for symbol early - it's not an object
    if val.is_symbol() {
        return Err("Symbol values are not supported".into());
    }

    // Check for Array before object (arrays are objects too)
    if val.is_array() {
        if depth >= MAX_UDF_VALUE_NESTING {
            return Err(udf_nesting_error());
        }
        let arr: Array = val.into_array().ok_or("Expected array")?;
        // Check for VecF32 marker
        if let Ok(ftype) = arr.as_object().get::<_, String>("__falkor_type")
            && ftype == "vecf32"
        {
            let mut vec = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                // Any f32 round-trips, including inf / NaN, which `vecf32()`
                // accepts too (as in C).
                let item: f64 = arr.get(i).map_err(|e| format!("VecF32 item error: {e}"))?;
                vec.push(item as f32);
            }
            return Ok(Value::VecF32(Arc::new(vec.into())));
        }
        let mut items = Vec::with_capacity(arr.len());
        for i in 0..arr.len() {
            let item: JsValue = arr.get(i).map_err(|e| format!("Array item error: {e}"))?;
            items.push(js_to_value_at(item, depth + 1)?);
        }
        return Ok(Value::List(Arc::new(items.into())));
    }

    if val.is_object() {
        let obj = val.as_object().unwrap();

        // Check for our custom classes via hidden __falkor_type property
        if let Ok(ftype) = obj.get::<_, String>("__falkor_type") {
            match ftype.as_str() {
                "node" => {
                    let id: u64 = obj.get("__falkor_node_id").map_err(|e| format!("{e}"))?;
                    return Ok(Value::Node(id.into()));
                }
                "edge" => {
                    let id: u64 = obj.get("__falkor_edge_id").map_err(|e| format!("{e}"))?;
                    return Ok(Value::Relationship(id.into()));
                }
                "path" => {
                    // Its members are converted recursively too.
                    if depth >= MAX_UDF_VALUE_NESTING {
                        return Err(udf_nesting_error());
                    }
                    let nodes_arr: Array = obj.get("nodes").map_err(|e| format!("{e}"))?;
                    let rels_arr: Array = obj.get("relationships").map_err(|e| format!("{e}"))?;
                    let n_nodes = nodes_arr.len();
                    let n_rels = rels_arr.len();
                    if n_rels != n_nodes.saturating_sub(1) {
                        return Err(format!(
                            "Invalid path: expected {} relationships for {} nodes, got {}",
                            n_nodes.saturating_sub(1),
                            n_nodes,
                            n_rels
                        ));
                    }
                    let mut path_values = Vec::with_capacity(n_nodes + n_rels);
                    for i in 0..n_nodes {
                        let node_val: JsValue = nodes_arr.get(i).map_err(|e| format!("{e}"))?;
                        let node = js_to_value_at(node_val, depth + 1)?;
                        if !matches!(node, Value::Node(_)) {
                            return Err(format!(
                                "Invalid path: element at node position {i} is not a Node"
                            ));
                        }
                        path_values.push(node);
                        if i < n_rels {
                            let rel_val: JsValue = rels_arr.get(i).map_err(|e| format!("{e}"))?;
                            let rel = js_to_value_at(rel_val, depth + 1)?;
                            if !matches!(rel, Value::Relationship(_)) {
                                return Err(format!(
                                    "Invalid path: element at relationship position {i} is not a Relationship"
                                ));
                            }
                            path_values.push(rel);
                        }
                    }
                    return Ok(Value::Path(Arc::new(path_values.into())));
                }
                "point" => {
                    let lat: f64 = obj.get("latitude").map_err(|e| format!("{e}"))?;
                    let lon: f64 = obj.get("longitude").map_err(|e| format!("{e}"))?;
                    return Ok(Value::Point(Point::new(lat as f32, lon as f32)));
                }
                _ => {}
            }
        }

        // Date / RegExp by their internal class, not by a `constructor`
        // property, which a plain object (e.g. a Cypher map) can carry too.
        let raw = obj.as_raw();
        // SAFETY: `raw` is a live value owned by `obj` for the duration of the call.
        if unsafe { rquickjs::qjs::JS_IsDate(raw) } {
            let get_time: rquickjs::Function = obj
                .get("getTime")
                .map_err(|e| format!("Date getTime error: {e}"))?;
            let ms: f64 = get_time
                .call((This(obj.clone()),))
                .map_err(|e| format!("Date getTime error: {e}"))?;
            if !ms.is_finite() {
                return Err(format!("Invalid Date value: {ms}"));
            }
            let secs = (ms / 1000.0) as i64;
            // Check the temporal type metadata to distinguish Date from Datetime
            if let Ok(tt) = obj.get::<_, String>("__falkor_temporal_type")
                && tt == "date"
            {
                return Ok(Value::Date(secs));
            }
            return Ok(Value::Datetime(secs));
        }
        // SAFETY: as above.
        if unsafe { rquickjs::qjs::JS_IsRegExp(raw) } {
            let to_string: rquickjs::Function = obj
                .get("toString")
                .map_err(|e| format!("RegExp toString error: {e}"))?;
            let s: String = to_string
                .call((This(obj.clone()),))
                .map_err(|e| format!("RegExp toString error: {e}"))?;
            return Ok(Value::String(Arc::new(s)));
        }

        // Plain object -> Map
        if depth >= MAX_UDF_VALUE_NESTING {
            return Err(udf_nesting_error());
        }
        // Filter out internal __falkor_* metadata, but keep escaped user keys (__falkor_esc_*)
        let keys: Vec<String> = obj
            .keys::<String>()
            .filter_map(Result::ok)
            .filter(|k| !k.starts_with("__falkor_") || k.starts_with("__falkor_esc_"))
            .collect();
        let mut map = Vec::with_capacity(keys.len());
        for key in keys {
            let v: JsValue = obj
                .get(&key)
                .map_err(|e| format!("Object get error: {e}"))?;
            // Unescape keys that were escaped during serialization
            let original_key = if key.starts_with("__falkor_esc_") {
                key.strip_prefix("__falkor_esc_").unwrap().to_string()
            } else {
                key
            };
            map.push((Arc::new(original_key), js_to_value_at(v, depth + 1)?));
        }
        return Ok(Value::Map(Arc::new(map.into_iter().collect())));
    }

    Err("Unsupported JS value type".to_string())
}

#[cfg(test)]
mod nesting_tests {
    use super::{MAX_UDF_VALUE_NESTING, js_to_value, udf_nesting_error, value_to_js};
    use crate::graph::{graph::Graph, graphblas::test_init::ensure_init};
    use crate::runtime::value::Value;
    use atomic_refcell::AtomicRefCell;
    use rquickjs::{Context, Runtime, Value as JsValue};
    use std::sync::Arc;
    use thin_vec::thin_vec;

    /// Run `f` in a fresh JS context on a thread with half the query pool's
    /// 2 MiB stack. The conversions grow their stack as they go, so a pass
    /// here does not depend on the build's frame sizes (ASAN's are ~6x).
    fn in_js_on_worker_stack(f: impl FnOnce(&rquickjs::Ctx<'_>) + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                let rt = Runtime::new().unwrap();
                let ctx = Context::full(&rt).unwrap();
                ctx.with(|ctx| f(&ctx));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// `depth` nested single-element arrays, built by a loop in JS.
    fn js_nested_array<'js>(
        ctx: &rquickjs::Ctx<'js>,
        depth: usize,
    ) -> JsValue<'js> {
        ctx.eval(format!(
            "(() => {{ let a = 1; for (let i = 0; i < {depth}; i++) a = [a]; return a; }})()"
        ))
        .unwrap()
    }

    fn list_depth(mut v: &Value) -> usize {
        let mut depth = 0;
        while let Value::List(items) = v {
            depth += 1;
            v = &items[0];
        }
        depth
    }

    fn nested(depth: usize) -> Value {
        let mut v = Value::Int(1);
        for _ in 0..depth {
            v = Value::List(Arc::new(thin_vec![v]));
        }
        v
    }

    #[test]
    fn js_array_at_the_cap_converts() {
        in_js_on_worker_stack(|ctx| {
            let v = js_to_value(js_nested_array(ctx, MAX_UDF_VALUE_NESTING)).unwrap();
            assert_eq!(list_depth(&v), MAX_UDF_VALUE_NESTING);
        });
    }

    #[test]
    fn js_array_past_the_cap_is_an_error() {
        in_js_on_worker_stack(|ctx| {
            let r = js_to_value(js_nested_array(ctx, MAX_UDF_VALUE_NESTING + 1));
            assert_eq!(r, Err(udf_nesting_error()));
        });
    }

    #[test]
    fn the_advisory_array_is_an_error() {
        // 300,000 levels, built in a loop: the UDF result that used to
        // exhaust the stack in the conversion.
        in_js_on_worker_stack(|ctx| {
            let r = js_to_value(js_nested_array(ctx, 300_000));
            assert_eq!(r, Err(udf_nesting_error()));
        });
    }

    #[test]
    fn a_self_containing_array_is_an_error() {
        in_js_on_worker_stack(|ctx| {
            let a: JsValue = ctx
                .eval("(() => { const a = []; a.push(a); return a; })()")
                .unwrap();
            assert_eq!(js_to_value(a), Err(udf_nesting_error()));
        });
    }

    #[test]
    fn js_objects_count_toward_the_cap() {
        in_js_on_worker_stack(|ctx| {
            let deep = |depth: usize| -> JsValue<'_> {
                ctx.eval(format!(
                    "(() => {{ let o = 1; for (let i = 0; i < {depth}; i++) o = {{k: o}}; return o; }})()"
                ))
                .unwrap()
            };
            assert!(matches!(
                js_to_value(deep(MAX_UDF_VALUE_NESTING)),
                Ok(Value::Map(_))
            ));
            assert_eq!(
                js_to_value(deep(MAX_UDF_VALUE_NESTING + 1)),
                Err(udf_nesting_error())
            );
        });
    }

    #[test]
    fn value_to_js_respects_the_cap() {
        ensure_init();
        in_js_on_worker_stack(|ctx| {
            let g = Arc::new(AtomicRefCell::new(Graph::new(16, 16, 1, 0, "udf_nesting")));
            // At the cap: converts, and converts back to the same value.
            let v = nested(MAX_UDF_VALUE_NESTING);
            let js = value_to_js(ctx, &v, &g, None).unwrap();
            // Compared by depth: `Value`'s `==` recurses too, and is not what
            // is under test.
            assert_eq!(list_depth(&js_to_value(js).unwrap()), MAX_UDF_VALUE_NESTING);
            // Past it: an error, not a stack overflow.
            assert_eq!(
                value_to_js(ctx, &nested(MAX_UDF_VALUE_NESTING + 1), &g, None).map(|_| ()),
                Err(udf_nesting_error())
            );
        });
    }

    #[test]
    fn ordinary_nesting_round_trips() {
        ensure_init();
        in_js_on_worker_stack(|ctx| {
            let g = Arc::new(AtomicRefCell::new(Graph::new(16, 16, 1, 0, "udf_nesting")));
            let v = Value::List(Arc::new(thin_vec![
                Value::Int(1),
                Value::List(Arc::new(thin_vec![Value::String(Arc::new("a".into()))])),
                Value::List(Arc::new(thin_vec![])),
            ]));
            let js = value_to_js(ctx, &v, &g, None).unwrap();
            assert_eq!(js_to_value(js).unwrap(), v);
        });
    }
}
