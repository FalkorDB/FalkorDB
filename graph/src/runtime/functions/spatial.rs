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
use thin_vec::ThinVec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "vecf32",
        args: [Type::Union(vec![
            Type::List(Box::new(Type::Any)),
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::VecF32, Type::Null]),
        fn vecf32(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
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
        args: [Type::Union(vec![Type::Map, Type::Null])],
        ret: Type::Union(vec![Type::Point, Type::Null]),
        fn point(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => {
                    let latitude = map
                        .get_str("latitude")
                        .ok_or_else(|| String::from("point() requires 'latitude' field"))?;
                    let latitude = match latitude {
                        Value::Float(f) => *f as f32,
                        Value::Int(i) => *i as f32,
                        _ => {
                            return Err(format!(
                                "Type mismatch: 'latitude' must be a number, got {}",
                                latitude.name()
                            ));
                        }
                    };
                    let longitude = map
                        .get_str("longitude")
                        .ok_or_else(|| String::from("point() requires 'longitude' field"))?;
                    let longitude = match longitude {
                        Value::Float(f) => *f as f32,
                        Value::Int(i) => *i as f32,
                        _ => {
                            return Err(format!(
                                "Type mismatch: 'longitude' must be a number, got {}",
                                longitude.name()
                            ));
                        }
                    };
                    let point = Point::new(latitude, longitude);
                    point.validate()?;
                    Ok(Value::Point(point))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "distance",
        args: [
            Type::Union(vec![Type::Point, Type::Null]),
            Type::Union(vec![Type::Point, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn distance(_, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(Value::Point(p1)), Some(Value::Point(p2))) => Ok(Value::Float(p1.distance(&p2))),
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "vec.euclideanDistance",
        args: [
            Type::Union(vec![Type::VecF32, Type::Null]),
            Type::Union(vec![Type::VecF32, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn vec_euclidean_distance(_, args) {
            let mut iter = args.into_iter();
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
                    Ok(Value::Float(vec_distance::euclidean(&a, &b).unwrap_or(0.0)))
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
            Type::Union(vec![Type::VecF32, Type::Null]),
            Type::Union(vec![Type::VecF32, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn vec_cosine_distance(_, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(Value::VecF32(a)), Some(Value::VecF32(b))) => {
                    if a.len() != b.len() {
                        return Err(format!(
                            "Vector dimension mismatch, expected {} but got {}",
                            a.len(), b.len()
                        ));
                    }
                    Ok(Value::Float(vec_distance::cosine(&a, &b).unwrap_or(1.0)))
                }
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );
}
