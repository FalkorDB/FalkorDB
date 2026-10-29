//! Spatial and vector functions.
//!
//! Functions for working with geographic points and dense float vectors.
//!
//! ```text
//!  Cypher                                Function                  Returns     Notes
//! ──────────────────────────────────────────────────────────────────────────────────
//!  vecf32(list)                          vecf32()                  VecF32      dense f32 vector from numeric list
//!  vec.euclideanDistance(v1, v2)         vec.euclideanDistance()   Float       L2 distance, sqrt(sum((a-b)^2))
//!  vec.cosineDistance(v1, v2)            vec.cosineDistance()      Float       1 - cos_similarity (matches RediSearch)
//!  point({lat, lon})                     point()                   Point       validates lat/lon ranges
//!  distance(p1, p2)                      distance()                Float       Haversine great-circle distance (m)
//! ```
//!
//! `point()` reads `latitude` and `longitude` fields from a Map and
//! constructs a validated `Point`.  `distance()` delegates to
//! `Point::distance()` which computes the Haversine formula.
//!
//! The two `vec.*` distance functions both:
//! - Propagate `Null` if either operand is `Null`.
//! - Reject inputs of mismatched dimension with `"Vector dimension mismatch"`.
//! - Are pinned by `tests/flow/test_vecsim.py::test01_vector_distance`.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use super::{FnType, Functions, Type};
use crate::runtime::{
    runtime::Runtime,
    value::{Point, Value},
    vec_distance,
};
use std::sync::Arc;

/// Build a point from the `latitude` / `longitude` entries of its map, with
/// C's checks and messages: exactly two keys, both present, both numeric, both
/// in range. `None` is a key the map lacks; `Some(Null)` is one set to `null`.
fn point_from_map_entries(
    key_count: usize,
    latitude: Option<&Value>,
    longitude: Option<&Value>,
) -> Result<Value, String> {
    if key_count != 2 {
        return Err(String::from(
            "A point map should have 2 elements, latitude and longitude",
        ));
    }
    let latitude = latitude.ok_or("Did not find 'latitude' value in point map")?;
    let longitude = longitude.ok_or("Did not find 'longitude' value in point map")?;
    let (Some(latitude), Some(longitude)) = (numeric(latitude), numeric(longitude)) else {
        return Err(String::from(
            "'latitude' and 'longitude' values in point map were not both valid numerics",
        ));
    };
    // C lets a NaN through (it fails neither side of the range test); a point is
    // kept a real coordinate here. An infinity fails the range test, as in C.
    // Ranges are checked before narrowing to `f32`, as C does, so a value just
    // past a bound is not rounded onto it.
    match (latitude, longitude) {
        (lat, _) if lat.is_nan() => Err(String::from("latitude must be a finite number, got NaN")),
        (_, lon) if lon.is_nan() => Err(String::from("longitude must be a finite number, got NaN")),
        (-90.0..=90.0, -180.0..=180.0) => {
            Ok(Value::Point(Point::new(latitude as f32, longitude as f32)))
        }
        (-90.0..=90.0, _) => Err(String::from(
            "longitude should be within the -180 to 180 range",
        )),
        _ => Err(String::from(
            "latitude should be within the -90 to 90 range",
        )),
    }
}

const fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "vecf32",
        args: [Type::union([
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::union([Type::VecF32, Type::Null]),
        fn vecf32(_, args) {
            match args.first() {
                Some(Value::List(vec)) => {
                    for v in vec.iter() {
                        if !matches!(v, Value::Int(_) | Value::Float(_)) {
                            return Err("vecf32 expects an array of numbers".to_string());
                        }
                    }
                    Ok(Value::VecF32(Arc::new(
                        vec.iter().map(|v| v.get_numeric() as f32).collect(),
                    )))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "point",
        args: [Type::union([Type::Map, Type::Null])],
        ret: Type::union([Type::Point, Type::Null]),
        fn point(_, args) {
            let mut iter = args.iter();
            match iter.next() {
                Some(Value::Map(map)) => point_from_map_entries(
                    map.len(),
                    map.get_str("latitude"),
                    map.get_str("longitude"),
                ),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "distance",
        args: [
            Type::union([Type::Point, Type::Null]),
            Type::union([Type::Point, Type::Null]),
        ],
        ret: Type::union([Type::Float, Type::Null]),
        fn distance(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::Point(p1)), Some(Value::Point(p2))) => Ok(Value::Float(p1.distance(p2))),
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "vec.euclideanDistance",
        args: [
            Type::union([Type::VecF32, Type::Null]),
            Type::union([Type::VecF32, Type::Null]),
        ],
        ret: Type::union([Type::Float, Type::Null]),
        fn vec_euclidean_distance(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::VecF32(a)), Some(Value::VecF32(b))) => {
                    if a.len() != b.len() {
                        return Err(format!(
                            "Vector dimension mismatch, expected {} but got {}",
                            a.len(), b.len()
                        ));
                    }
                    // SIMD path delegates to `simsimd` (AVX-512 / AVX2 /
                    // NEON / scalar runtime dispatch). `None` only on
                    // length mismatch, ruled out above.
                    Ok(Value::Float(vec_distance::euclidean(a.as_slice(), b.as_slice()).unwrap_or(0.0)))
                }
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    // Cosine *distance* — `1 - cosine_similarity`. Matches RediSearch's
    // `COSINE` metric so the value the user sees from `vec.cosineDistance`
    // and from `db.idx.vector.queryNodes(... COSINE ...)` agree.
    cypher_fn!(funcs, "vec.cosineDistance",
        args: [
            Type::union([Type::VecF32, Type::Null]),
            Type::union([Type::VecF32, Type::Null]),
        ],
        ret: Type::union([Type::Float, Type::Null]),
        fn vec_cosine_distance(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::VecF32(a)), Some(Value::VecF32(b))) => {
                    if a.len() != b.len() {
                        return Err(format!(
                            "Vector dimension mismatch, expected {} but got {}",
                            a.len(), b.len()
                        ));
                    }
                    Ok(Value::Float(vec_distance::cosine(a.as_slice(), b.as_slice()).unwrap_or(1.0)))
                }
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );
}
