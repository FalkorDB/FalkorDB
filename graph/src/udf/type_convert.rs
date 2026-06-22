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
                let js_val = value_to_js(ctx, item, graph, runtime)?;
                arr.set(i, js_val)
                    .map_err(|e| format!("JS array set error: {e}"))?;
            }
            Ok(arr.into_value())
        }
        Value::Map(map) => {
            let obj =
                Object::new(ctx.clone()).map_err(|e| format!("JS object creation error: {e}"))?;
            for (key, val) in map.iter() {
                let js_val = value_to_js(ctx, val, graph, runtime)?;
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
            crate::udf::js_classes::create_js_node(ctx, (*node_id).into(), graph)
        }
        Value::Relationship(rel_id) => {
            let (src, dst) = match runtime {
                Some(rt) => rt.get_relationship_endpoints(*rel_id),
                None => graph.borrow().get_relationship_endpoints(*rel_id),
            };
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
        if f == f.floor() && f.abs() < (1i64 << 53) as f64 {
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
        let arr: Array = val.into_array().ok_or("Expected array")?;
        // Check for VecF32 marker
        if let Ok(ftype) = arr.as_object().get::<_, String>("__falkor_type")
            && ftype == "vecf32"
        {
            let mut vec = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                let item: f64 = arr.get(i).map_err(|e| format!("VecF32 item error: {e}"))?;
                if !item.is_finite() {
                    return Err(format!("VecF32 element at index {i} is not finite"));
                }
                vec.push(item as f32);
            }
            return Ok(Value::VecF32(Arc::new(vec.into())));
        }
        let mut items = Vec::with_capacity(arr.len());
        for i in 0..arr.len() {
            let item: JsValue = arr.get(i).map_err(|e| format!("Array item error: {e}"))?;
            items.push(js_to_value(item)?);
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
                        let node = js_to_value(node_val)?;
                        if !matches!(node, Value::Node(_)) {
                            return Err(format!(
                                "Invalid path: element at node position {i} is not a Node"
                            ));
                        }
                        path_values.push(node);
                        if i < n_rels {
                            let rel_val: JsValue = rels_arr.get(i).map_err(|e| format!("{e}"))?;
                            let rel = js_to_value(rel_val)?;
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

        // Check constructor name for Date/RegExp
        if let Ok(constructor) = obj.get::<_, Object>("constructor")
            && let Ok(name) = constructor.get::<_, String>("name")
        {
            match name.as_str() {
                "Date" => {
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
                "RegExp" => {
                    let to_string: rquickjs::Function = obj
                        .get("toString")
                        .map_err(|e| format!("RegExp toString error: {e}"))?;
                    let s: String = to_string
                        .call((This(obj.clone()),))
                        .map_err(|e| format!("RegExp toString error: {e}"))?;
                    return Ok(Value::String(Arc::new(s)));
                }
                _ => {}
            }
        }

        // Plain object -> Map
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
            map.push((Arc::new(original_key), js_to_value(v)?));
        }
        return Ok(Value::Map(Arc::new(map.into_iter().collect())));
    }

    Err("Unsupported JS value type".to_string())
}
