//! The geo kind: `Point` values, the native replacement for the GEO half of a RediSearch Range
//! field. It answers `distance(n.loc, point(...)) < r`.
//!
//! # The key
//!
//! A point is two `f32`s, which is exactly 64 bits — so the key is a **lossless** bijection, not a
//! quantisation. Each coordinate goes through the same monotone `f32 -> u32` transform the numeric
//! kind uses on `f64`, and the two 32-bit images are bit-interleaved into a Morton (Z-order) code:
//!
//! ```text
//!   key = spread(enc(lat)) | spread(enc(lon)) << 1
//! ```
//!
//! Losslessness buys exactness on `n.loc = point(...)`: the key identifies the point, so equality
//! is a plain point lookup and needs no re-check. Interleaving buys locality: any rectangle in
//! (lat, lon) is a union of Morton intervals, because the code is a depth-32 quadtree address and
//! a quadtree cell's codes are contiguous.
//!
//! # The search
//!
//! A radius search is a circle, and a circle is not a rectangle, so [`cover`] answers with the
//! quadtree cells overlapping the circle's bounding box — a **superset** of the true answer. That
//! is sound here because `utilize_index` always retains the filter for a distance predicate
//! (`distance(...)` is a function invocation, which `needs_post_filter` treats as unresolvable), so
//! the runtime rechecks the true great-circle distance on every candidate row. The bound on the
//! number of intervals is a cost knob, not a correctness one: fewer intervals means a coarser
//! cover and more rows for the filter to reject.
//!
//! # What comes from the `geo` crate
//!
//! The geodesy and the rectangle algebra are the crate's, not ours:
//!
//! * [`HaversineMeasure`] models the sphere and [`Destination`] walks the latitude extremes of the
//!   search circle. The measure is constructed on **FalkorDB's** radius, so the box is drawn on the
//!   same sphere `distance()` is evaluated on — a box drawn on a smaller sphere would clip the
//!   circle and lose rows, which the retained filter could never put back.
//! * [`Rect`], [`Intersects`] and [`Contains`] decide cell-versus-query overlap and containment.
//!   `Rect<u32>` is used over the *encoded* grid rather than over degrees: the encoding is monotone
//!   per axis, so a rectangle in one space is a rectangle in the other, and integer comparisons are
//!   exact where float ones would need an epsilon.
//!
//! What the crate has no answer for is the key itself: Z-order interleaving is not in `geo` (nor is
//! any persistent spatial index — `rstar`, which `geo-types` pulls in, is neither copy-on-write nor
//! `(key, doc)`-shaped, so it cannot back an MVCC column). Those parts stay here.

use geo::{Contains, Destination, HaversineMeasure, Intersects, Point as GeoPoint, Rect, coord};

use super::doc_iter::{DocIter, Tree, UnionIter, empty_docs};
use crate::runtime::value::{Point, Value};

/// The sphere the bounding box is drawn on: FalkorDB's earth radius, the one
/// [`Point::distance`](crate::runtime::value::Point::distance) uses. Passing it to `geo` rather
/// than taking `geo`'s own `Haversine` (a GRS80 mean radius of 6 371 008.8 m) keeps the box and the
/// predicate on one model — otherwise the two disagree by ~0.1%, and the disagreement is in rows.
const FALKORDB_SPHERE: HaversineMeasure = HaversineMeasure::new(EARTH_RADIUS_M);

/// Earth radius used for the bounding box, in metres. The same value
/// [`Point::distance`](crate::runtime::value::Point::distance) uses — a larger one here would be
/// harmless (a wider box), a smaller one would clip the circle and lose rows.
const EARTH_RADIUS_M: f64 = 6_378_140.0;

/// Ceiling on the intervals one cover may produce. Reaching it stops the refinement and emits the
/// cell whole, which widens the superset rather than dropping anything.
const MAX_RANGES: usize = 32;

/// IEEE-754 sign bit of an `f32`.
const SIGN32: u32 = 0x8000_0000;

/// A geo property index over one `(label, attribute)`: entity ids keyed by the Morton code of
/// their point.
///
/// `Clone` is `O(1)` — a root-`Arc` bump. There is no array key space: RediSearch does not index
/// point elements inside a list either, and no Cypher predicate reaches for one.
#[derive(Clone, Default)]
pub struct GeoIndex {
    tree: Tree,
}

impl GeoIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the index holds no tuples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Index `id` under `value`. A no-op for anything that is not a point, and for a point with a
    /// `NaN` coordinate — which has no position to index it at.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        if let Some(k) = Self::key_of(value) {
            let _newly_inserted = self.tree.insert(k, id);
        }
    }

    /// Remove `id` from under `value`. A no-op if it was never indexed.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        if let Some(k) = Self::key_of(value) {
            let _was_present = self.tree.remove(k, id);
        }
    }

    /// Encode `(value, id)` entries to `(key, doc)` tuples, dropping non-point values.
    #[must_use]
    pub fn encode_entries(entries: &[(Value, u64)]) -> Vec<(u64, u64)> {
        entries
            .iter()
            .filter_map(|(v, id)| Self::key_of(v).map(|k| (k, *id)))
            .collect()
    }

    /// Bulk-build from `(value, id)` pairs in any order.
    #[must_use]
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a Value, u64)>) -> Self {
        let tuples = entries
            .into_iter()
            .filter_map(|(v, id)| Self::key_of(v).map(|k| (k, id)))
            .collect();
        Self::from_encoded(tuples)
    }

    /// Build directly from already-encoded tuples — the install adopting a background-built BASE.
    #[must_use]
    pub fn from_encoded(mut tuples: Vec<(u64, u64)>) -> Self {
        tuples.sort_unstable();
        tuples.dedup();
        Self {
            tree: Tree::from_sorted(&tuples),
        }
    }

    /// Every `(key, doc)` tuple, in key order — the install's DELTA/TOMB enumeration.
    #[must_use]
    pub fn encoded_tuples(&self) -> Vec<(u64, u64)> {
        self.tree.range_tuples(0, u64::MAX).collect()
    }

    /// Add already-encoded tuples (install: replay DELTA onto BASE).
    pub fn add_encoded(
        &mut self,
        tuples: &mut [(u64, u64)],
    ) {
        tuples.sort_unstable();
        self.tree.insert_batch(tuples);
    }

    /// Remove already-encoded tuples (install: subtract TOMB from BASE).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut [(u64, u64)],
    ) {
        tuples.sort_unstable();
        self.tree.remove_batch(tuples);
    }

    /// Entity ids whose point equals `value` exactly. The key is a bijection, so this needs no
    /// re-check.
    #[must_use]
    pub fn point(
        &self,
        value: &Value,
    ) -> DocIter {
        match Self::key_of(value) {
            Some(k) => DocIter::One(self.tree.point(k)),
            None => empty_docs(&self.tree),
        }
    }

    /// Entity ids within `radius_m` metres of `centre` — as a **superset**: every id whose point
    /// lies in a quadtree cell that overlaps the circle's bounding box. The caller's retained
    /// filter rechecks the true distance.
    ///
    /// A non-positive or non-finite radius selects nothing.
    #[must_use]
    pub fn within(
        &self,
        centre: &Point,
        radius_m: f64,
    ) -> DocIter {
        let Some(rect) = bounding_rect(centre, radius_m) else {
            return empty_docs(&self.tree);
        };
        DocIter::Many(UnionIter::new(
            vec![self.tree.clone()],
            cover(&rect)
                .into_iter()
                .map(|(lo, hi)| (0, lo, hi))
                .collect(),
        ))
    }

    /// The tree — for the column facade, which composes windows across kinds.
    pub(super) fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The Morton key of a point value, or `None` for any other value (and for a point carrying a
    /// `NaN` coordinate, which has no position in a sorted index — the same rule the numeric kind
    /// applies to a `NaN` number).
    pub(super) fn key_of(value: &Value) -> Option<u64> {
        match value {
            Value::Point(p) if p.latitude.is_nan() || p.longitude.is_nan() => None,
            Value::Point(p) => Some(morton(enc_f32(p.latitude), enc_f32(p.longitude))),
            _ => None,
        }
    }
}

/// Monotone total order over non-`NaN` `f32`, as a `u32` — the `f32` twin of
/// [`encode_f64`](super::encode::encode_f64), and monotone for the same reason: non-negatives get
/// the sign bit set so they sort above every negative, negatives are bit-inverted so
/// more-negative sorts lower, and `-0.0` collapses into `+0.0`.
#[must_use]
fn enc_f32(x: f32) -> u32 {
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    if bits & SIGN32 == 0 {
        bits | SIGN32
    } else {
        !bits
    }
}

/// Spread the 32 bits of `x` into the even positions of a `u64` (`b31..b0` -> `b62,b60,..,b0`).
#[must_use]
fn spread(x: u32) -> u64 {
    let mut v = u64::from(x);
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    v = (v | (v << 1)) & 0x5555_5555_5555_5555;
    v
}

/// The Morton code of an encoded coordinate pair: latitude in the even bits, longitude in the odd.
#[must_use]
fn morton(
    lat: u32,
    lon: u32,
) -> u64 {
    spread(lat) | (spread(lon) << 1)
}

/// A rectangle over the *encoded* grid: `x` is the encoded longitude, `y` the encoded latitude,
/// following `geo`'s convention. Inclusive on every side, which is what `geo`'s [`Intersects`] and
/// [`Contains`] mean for a [`Rect`] anyway.
type GridRect = Rect<u32>;

/// The rectangle in encoded space with these inclusive corners.
fn grid_rect(
    lat_lo: u32,
    lat_hi: u32,
    lon_lo: u32,
    lon_hi: u32,
) -> GridRect {
    Rect::new(
        coord! { x: lon_lo, y: lat_lo },
        coord! { x: lon_hi, y: lat_hi },
    )
}

/// The encoded rectangle bounding the great-circle disk of `radius_m` around `centre`, or `None`
/// when the disk is empty or undefined (a non-finite or non-positive radius, a `NaN` centre).
///
/// The latitude edges are walked by `geo` on FalkorDB's sphere — the extremes of a circle *are* due
/// north and due south of its centre, so two [`Destination`] calls give them exactly.
///
/// Longitude is not a `destination` call, and the difference is a correctness one rather than a
/// stylistic one: the easternmost point of a spherical cap is **not** the one due east of the
/// centre, it is where the circle runs tangent to a meridian, further out. Walking east would draw
/// a box narrower than the circle and silently lose rows. The tangent longitude has a closed form,
/// `Δλ = asin(sin ρ / cos φ)` for angular radius `ρ`, which is what is used; a cap reaching a pole
/// (or wrapping the antimeridian) takes the whole longitude range instead of being split in two.
fn bounding_rect(
    centre: &Point,
    radius_m: f64,
) -> Option<GridRect> {
    if !radius_m.is_finite()
        || radius_m <= 0.0
        || centre.latitude.is_nan()
        || centre.longitude.is_nan()
    {
        return None;
    }
    let lat = f64::from(centre.latitude);
    let lon = f64::from(centre.longitude);
    let origin = GeoPoint::new(lon, lat);
    let rho_deg = (radius_m / EARTH_RADIUS_M).to_degrees(); // angular radius

    // A cap that reaches a pole is a special case for both axes, and getting it wrong is not
    // conservative: `destination` walking north past the pole comes back down the far side and
    // reports a *lower* latitude than the cap actually spans, and the longitude formula below
    // folds back on itself once the angular radius passes 90°. Both would draw a box inside the
    // circle and lose rows, so the pole cases are answered before either is consulted.
    let reaches_north = lat + rho_deg >= 90.0;
    let reaches_south = lat - rho_deg <= -90.0;

    // Otherwise: due north and due south are the latitude extremes, walked by `geo` on the same
    // sphere `distance()` uses.
    let lat_hi = if reaches_north {
        90.0
    } else {
        FALKORDB_SPHERE.destination(origin, 0.0, radius_m).y()
    };
    let lat_lo = if reaches_south {
        -90.0
    } else {
        FALKORDB_SPHERE.destination(origin, 180.0, radius_m).y()
    };

    let sin_ratio = rho_deg.to_radians().sin() / lat.to_radians().cos();
    let (lon_lo, lon_hi) = if reaches_north
        || reaches_south
        || rho_deg >= 90.0
        || !(-1.0..=1.0).contains(&sin_ratio)
    {
        (-180.0, 180.0) // a pole, or more than a hemisphere: every meridian is in range
    } else {
        let d_lon = sin_ratio.asin().to_degrees();
        if lon - d_lon < -180.0 || lon + d_lon > 180.0 {
            (-180.0, 180.0) // wraps the antimeridian: take the lot rather than split
        } else {
            (lon - d_lon, lon + d_lon)
        }
    };

    // The box is computed in `f64` and compared against `f32` keys, so the narrowing cast can
    // round an edge *inwards* and clip a point that genuinely lies on it. Widen each edge by a
    // relative slack well above the `f32` step before casting: the box is already a superset, and
    // a hair more of it costs the filter a row it would have rejected anyway.
    let pad = |v: f64, dir: f64| (v + dir * v.abs().mul_add(1e-6, 1e-6)) as f32;
    Some(grid_rect(
        enc_f32(pad(lat_lo, -1.0)),
        enc_f32(pad(lat_hi, 1.0)),
        enc_f32(pad(lon_lo, -1.0)),
        enc_f32(pad(lon_hi, 1.0)),
    ))
}

/// The Morton intervals covering `q`: a set of **disjoint** quadtree cells whose union contains
/// every point of the rectangle (and, at the edges, some points outside it).
///
/// Disjointness is what lets the caller chain the intervals without deduplicating: a point lies in
/// exactly one cell of the decomposition, so no doc is yielded twice.
fn cover(q: &GridRect) -> Vec<(u64, u64)> {
    /// A quadtree cell: the corner of its `2^bits` square, in encoded coordinates.
    #[derive(Clone, Copy)]
    struct Cell {
        lat_base: u32,
        lon_base: u32,
        bits: u32,
    }

    impl Cell {
        /// The cell as a rectangle over the encoded grid, so `geo` can answer the overlap and
        /// containment questions instead of a hand-rolled corner comparison.
        fn rect(self) -> GridRect {
            let span = if self.bits >= 32 {
                u32::MAX
            } else {
                (1u32 << self.bits) - 1
            };
            grid_rect(
                self.lat_base,
                self.lat_base | span,
                self.lon_base,
                self.lon_base | span,
            )
        }

        /// The contiguous Morton interval this cell's points occupy.
        fn interval(self) -> (u64, u64) {
            let min = morton(self.lat_base, self.lon_base);
            let max = if self.bits >= 32 {
                u64::MAX
            } else {
                min | ((1u64 << (2 * self.bits)) - 1)
            };
            (min, max)
        }
    }

    // **Largest cell first**, not depth-first. A depth-first walk spends its whole budget on the
    // first corner of the box and then has to emit whatever cell it is standing in — near the root
    // that cell is a quarter of the planet, so a 20km search would return a continent. Splitting
    // the biggest overlapping cell each round instead keeps the coarse cells from surviving: the
    // budget is spent where it removes the most area.
    let mut cells = vec![Cell {
        lat_base: 0,
        lon_base: 0,
        bits: 32,
    }];
    if !cells[0].rect().intersects(q) {
        return Vec::new();
    }
    // Bounded so a degenerate query cannot spin: every round either splits a cell (at most 32
    // levels deep per branch) or stops.
    for _ in 0..1024 {
        // The largest cell that is not already wholly inside the box — the one worth refining.
        let Some(i) = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.bits > 0 && !q.contains(&c.rect()))
            .max_by_key(|(_, c)| c.bits)
            .map(|(i, _)| i)
        else {
            break; // every cell is inside the box, or is a single point
        };
        // Splitting replaces one cell with up to four. Stop before overrunning the budget: the
        // cover stays valid, just coarser.
        if cells.len() + 3 > MAX_RANGES {
            break;
        }
        let c = cells.swap_remove(i);
        let bits = c.bits - 1;
        let step = 1u32 << bits;
        for (dlat, dlon) in [(0, 0), (0, step), (step, 0), (step, step)] {
            let child = Cell {
                lat_base: c.lat_base + dlat,
                lon_base: c.lon_base + dlon,
                bits,
            };
            if child.rect().intersects(q) {
                cells.push(child);
            }
        }
    }

    let mut out: Vec<(u64, u64)> = cells.into_iter().map(Cell::interval).collect();
    // Coalesce: the decomposition emits siblings separately, and four sibling cells that all
    // survived are one contiguous interval. Merging them turns four cursor descents into one.
    out.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(out.len());
    for (lo, hi) in out {
        match merged.last_mut() {
            Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(
        lat: f32,
        lon: f32,
    ) -> Value {
        Value::Point(Point::new(lat, lon))
    }

    fn ids(it: DocIter) -> Vec<u64> {
        let mut v: Vec<u64> = it.collect();
        v.sort_unstable();
        v
    }

    /// The key is a bijection on non-`NaN` points: distinct points never share a key, so equality
    /// is exact without a re-check.
    #[test]
    fn the_key_is_lossless() {
        let pts = [
            (0.0f32, 0.0f32),
            (-0.0, 0.0),
            (0.0, -0.0),
            (51.5074, -0.1278),
            (-33.8688, 151.2093),
            (90.0, 180.0),
            (-90.0, -180.0),
            (f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
        ];
        let mut seen = std::collections::HashMap::new();
        for &(lat, lon) in &pts {
            let k = GeoIndex::key_of(&p(lat, lon)).unwrap();
            // -0.0 and +0.0 are the same coordinate and must share a key; everything else differs.
            let canonical = (
                if lat == 0.0 { 0.0f32 } else { lat }.to_bits(),
                if lon == 0.0 { 0.0f32 } else { lon }.to_bits(),
            );
            if let Some(prev) = seen.insert(k, canonical) {
                assert_eq!(prev, canonical, "two distinct points collided on key {k}");
            }
        }
        assert_eq!(GeoIndex::key_of(&p(f32::NAN, 0.0)), None);
        assert_eq!(GeoIndex::key_of(&Value::Int(1)), None);
    }

    /// A cover must contain every point of its rectangle. Checked exhaustively against a brute
    /// force over a grid of points: anything inside the box must fall in some interval.
    #[test]
    fn a_cover_contains_its_rectangle() {
        let centre = Point::new(51.5, -0.12);
        let rect = bounding_rect(&centre, 5_000.0).expect("a real radius has a box");
        let ranges = cover(&rect);
        assert!(!ranges.is_empty() && ranges.len() <= MAX_RANGES);

        for i in 0..40 {
            for j in 0..40 {
                let lat = 51.5 + (f64::from(i) - 20.0) * 0.002;
                let lon = -0.12 + (f64::from(j) - 20.0) * 0.002;
                let (elat, elon) = (enc_f32(lat as f32), enc_f32(lon as f32));
                let inside = rect.contains(&coord! { x: elon, y: elat });
                let key = morton(elat, elon);
                let covered = ranges.iter().any(|&(lo, hi)| key >= lo && key <= hi);
                assert!(
                    !inside || covered,
                    "({lat}, {lon}) is in the box but in no interval"
                );
            }
        }
    }

    /// The intervals are disjoint, which is what lets the union chain them without deduplicating.
    #[test]
    fn cover_intervals_are_disjoint() {
        for radius in [1.0, 100.0, 10_000.0, 1_000_000.0, 20_000_000.0] {
            let rect = bounding_rect(&Point::new(12.5, 77.5), radius).unwrap();
            let ranges = cover(&rect);
            for w in ranges.windows(2) {
                assert!(w[0].1 < w[1].0, "overlapping intervals at radius {radius}");
            }
        }
    }

    /// The property the whole kind exists for: a radius search returns every point inside the
    /// circle. It may return more (the box, and a coarse cover of it) — the retained filter
    /// rejects those — but it may never miss one.
    #[test]
    fn within_returns_every_point_inside_the_circle() {
        let centre = Point::new(40.7128, -74.0060); // New York
        let mut idx = GeoIndex::new();
        let mut expected = Vec::new();
        let mut id = 0u64;
        for i in -10..=10 {
            for j in -10..=10 {
                let pt = Point::new(
                    centre.latitude + i as f32 * 0.01,
                    centre.longitude + j as f32 * 0.01,
                );
                if pt.distance(&centre) <= 1_000.0 {
                    expected.push(id);
                }
                idx.add(&Value::Point(pt), id);
                id += 1;
            }
        }
        assert!(!expected.is_empty(), "the fixture must have hits at all");

        let got = ids(idx.within(&centre, 1_000.0));
        for e in &expected {
            assert!(
                got.contains(e),
                "id {e} is inside the circle but was missed"
            );
        }
        assert!(
            got.len() < id as usize,
            "a radius search must narrow the candidate set, not return everything"
        );
    }

    /// Far-away points are not returned, and the degenerate radii answer empty rather than
    /// everything.
    #[test]
    fn within_excludes_the_far_side_of_the_world_and_rejects_bad_radii() {
        let mut idx = GeoIndex::new();
        idx.add(&p(51.5074, -0.1278), 1); // London
        idx.add(&p(-33.8688, 151.2093), 2); // Sydney

        let london = Point::new(51.5074, -0.1278);
        assert_eq!(ids(idx.within(&london, 10_000.0)), vec![1]);
        assert!(ids(idx.within(&london, 0.0)).is_empty());
        assert!(ids(idx.within(&london, -1.0)).is_empty());
        assert!(ids(idx.within(&london, f64::NAN)).is_empty());
        // Half the planet reaches both.
        assert_eq!(ids(idx.within(&london, 20_000_000.0)), vec![1, 2]);
    }

    /// The bounding box and the predicate must be evaluated on the SAME sphere. `geo`'s own
    /// `Haversine` is a GRS80 mean radius, ~7km smaller than FalkorDB's; on that sphere a given
    /// distance subtends a larger angle, so a box drawn with it would be *wider* — harmless. The
    /// dangerous direction is the reverse, and this pins the two together so a future edit to
    /// either radius fails here rather than in a query.
    #[test]
    fn the_box_is_drawn_on_the_engines_sphere() {
        use geo::Distance;

        for (a, b) in [
            ((51.5074f32, -0.1278f32), (48.8566f32, 2.3522f32)),
            ((0.0, 0.0), (0.0, 1.0)),
            ((-33.8688, 151.2093), (35.6762, 139.6503)),
        ] {
            let engine = Point::new(a.0, a.1).distance(&Point::new(b.0, b.1));
            let crate_side = FALKORDB_SPHERE.distance(
                GeoPoint::new(f64::from(a.1), f64::from(a.0)),
                GeoPoint::new(f64::from(b.1), f64::from(b.0)),
            );
            let rel = (engine - crate_side).abs() / engine;
            assert!(
                rel < 1e-6,
                "the sphere the box is drawn on ({crate_side} m) must be the one distance() uses \
                 ({engine} m)"
            );
        }
    }

    /// A cap that reaches a pole, or spans more than a hemisphere, cannot use either the
    /// `destination` walk (it comes back down the far side) or the tangent-longitude formula (it
    /// folds back past 90°). Both failures are the bad direction — a box *inside* the circle.
    #[test]
    fn polar_and_hemispheric_radii_take_the_whole_range() {
        let mut idx = GeoIndex::new();
        idx.add(&p(89.9, 100.0), 1); // near the north pole, far side of the world in longitude
        idx.add(&p(-45.0, -170.0), 2);
        idx.add(&p(0.0, 0.0), 3);

        // A 1000km circle around the pole must find the point beside it, whatever its longitude.
        let pole = Point::new(90.0, 0.0);
        assert_eq!(ids(idx.within(&pole, 1_000_000.0)), vec![1]);

        // More than a hemisphere reaches everything.
        assert_eq!(
            ids(idx.within(&Point::new(0.0, 0.0), 20_000_000.0)),
            vec![1, 2, 3]
        );

        // And a circle that wraps the antimeridian still finds what is on the other side of it.
        assert_eq!(
            ids(idx.within(&Point::new(-45.0, 179.0), 1_000_000.0)),
            vec![2]
        );
    }

    /// Equality on a point is exact — the key is a bijection.
    #[test]
    fn point_equality_is_exact() {
        let mut idx = GeoIndex::new();
        idx.add(&p(1.0, 2.0), 1);
        idx.add(&p(1.0, 2.000_001), 2);
        assert_eq!(ids(idx.point(&p(1.0, 2.0))), vec![1]);
        assert_eq!(ids(idx.point(&p(1.0, 2.000_001))), vec![2]);
        assert!(ids(idx.point(&p(2.0, 1.0))).is_empty());
    }

    #[test]
    fn remove_and_bulk_build_agree_with_incremental() {
        let entries = [(p(1.0, 2.0), 1u64), (p(3.0, 4.0), 2), (Value::Int(9), 3)];
        let bulk = GeoIndex::from_entries(entries.iter().map(|(v, id)| (v, *id)));
        let mut inc = GeoIndex::new();
        for (v, id) in &entries {
            inc.add(v, *id);
        }
        assert_eq!(bulk.encoded_tuples(), inc.encoded_tuples());

        let mut idx = inc;
        idx.remove(&p(1.0, 2.0), 1);
        assert!(ids(idx.point(&p(1.0, 2.0))).is_empty());
        idx.remove(&p(3.0, 4.0), 2);
        assert!(idx.is_empty());
    }
}
