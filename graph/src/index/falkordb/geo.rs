//! The geo kind: `Point` values, the native replacement for the GEO half of a RediSearch Range
//! field. It answers `distance(n.loc, point(...)) < r`.
//!
//! # The key
//!
//! Each coordinate is quantised **uniformly** onto a `u32` — latitude across `[-90, 90]`, longitude
//! across `[-180, 180]` — and the two grids are bit-interleaved into a Morton (Z-order) code:
//!
//! ```text
//!   key = spread(quantise(lat)) | spread(quantise(lon)) << 1
//! ```
//!
//! Uniformity is the whole point, and it is worth saying why the obvious alternative fails. A point
//! is two `f32`s, which is exactly 64 bits, so interleaving the *bit patterns* gives a lossless
//! key — and a useless index. Float spacing is exponential: a 10 km box straddling the origin spans
//! nearly half of that encoded axis, because every value between 1e-38 and 0.09 lives in there.
//! Measured on the bench's `distance index scan geo` corpus, the lossless key returned **100
//! candidates for 7 real hits** — a full scan wearing an index's clothes. The uniform grid puts
//! 2³² steps across 180°, so a cell is a real square on the ground (~4.7 mm at the equator) and the
//! same query narrows to its neighbourhood.
//!
//! What uniformity costs is exactness: two points within one grid step share a key, so
//! `n.loc = point(...)` answers a **superset**. That is safe here for the same reason the radius
//! search is — a point operand is always a `point(...)` call, a parameter, or a variable, and
//! `needs_post_filter` treats all three as unresolvable, so the filter is retained and the runtime
//! rechecks. A `Point` cannot be written as a bare literal in Cypher, so there is no fourth case.
//!
//! Interleaving buys locality: any rectangle in (lat, lon) is a union of Morton intervals, because
//! the code is a depth-32 quadtree address and a quadtree cell's codes are contiguous.
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

/// The quantisation grid: latitude spans 180°, longitude 360°, each across the whole `u32` range.
/// One step is ~4.2e-8° — about 4.7 mm of latitude.
const LAT_LO: f64 = -90.0;
const LAT_SPAN: f64 = 180.0;
const LON_LO: f64 = -180.0;
const LON_SPAN: f64 = 360.0;

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

    /// Entity ids whose point quantises to the same grid cell as `value` — a **superset** of the
    /// points equal to it, narrowed by the caller's retained filter (see the module docs on why a
    /// point operand always keeps one).
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
            Value::Point(p) => Some(morton(
                quantise(f64::from(p.latitude), LAT_LO, LAT_SPAN),
                quantise(f64::from(p.longitude), LON_LO, LON_SPAN),
            )),
            _ => None,
        }
    }
}

/// A degree coordinate on its uniform `u32` grid. Monotone, and saturating rather than wrapping:
/// a coordinate outside its range (which the parser rejects, but a decoded RDB might not) pins to
/// an edge of the grid instead of folding round to the far side of the planet.
#[must_use]
fn quantise(
    v: f64,
    lo: f64,
    span: f64,
) -> u32 {
    let t = ((v - lo) / span).clamp(0.0, 1.0);
    // `round` rather than `floor`: the grid step is the quantisation error either way, and
    // rounding halves its magnitude.
    (t * f64::from(u32::MAX)).round() as u32
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

    // Quantisation rounds to the nearest grid step, so an edge can land one step inside the box
    // and clip a point sitting exactly on it. Widen by that one step: the box is already a
    // superset, and 4.7 mm more of it costs the filter a row it would have rejected anyway.
    Some(grid_rect(
        quantise(lat_lo, LAT_LO, LAT_SPAN).saturating_sub(1),
        quantise(lat_hi, LAT_LO, LAT_SPAN).saturating_add(1),
        quantise(lon_lo, LON_LO, LON_SPAN).saturating_sub(1),
        quantise(lon_hi, LON_LO, LON_SPAN).saturating_add(1),
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
    // that cell is a quarter of the planet, so a 20km search would return a continent. Refining the
    // biggest overlapping cell first spends the budget where it removes the most area.
    //
    // Breadth-first *is* largest-first here, and for free: a child is always one level smaller than
    // its parent, so a FIFO hands cells back in non-increasing size. The alternative — rescanning a
    // list for its largest member each round — re-tests every surviving cell against the query on
    // every round, which measured as the bulk of a small radius search's cost.
    let root = Cell {
        lat_base: 0,
        lon_base: 0,
        bits: 32,
    };
    if !root.rect().intersects(q) {
        return Vec::new();
    }
    let mut queue = std::collections::VecDeque::from([root]);
    let mut cells: Vec<Cell> = Vec::with_capacity(MAX_RANGES);
    while let Some(c) = queue.pop_front() {
        // Final: wholly inside the box, or a single grid point — nothing to gain by splitting.
        // Also final once the budget is spent: the cover stays valid, just coarser.
        if c.bits == 0 || q.contains(&c.rect()) || cells.len() + queue.len() + 4 > MAX_RANGES {
            cells.push(c);
            continue;
        }
        let bits = c.bits - 1;
        let step = 1u32 << bits;
        for (dlat, dlon) in [(0, 0), (0, step), (step, 0), (step, step)] {
            let child = Cell {
                lat_base: c.lat_base + dlat,
                lon_base: c.lon_base + dlon,
                bits,
            };
            if child.rect().intersects(q) {
                queue.push_back(child);
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
    /// Distinct points get distinct keys down to the grid step, and the grid step is small enough
    /// that no realistic pair of places collides. Points closer than ~5 mm share a cell — which is
    /// why equality answers a superset and the filter is retained.
    #[test]
    fn the_grid_separates_distinct_places() {
        let pts = [
            (0.0f32, 0.0f32),
            (51.5074, -0.1278),
            (-33.8688, 151.2093),
            (90.0, 180.0),
            (-90.0, -180.0),
            (0.000_01, 0.000_01),
        ];
        let mut seen = std::collections::HashMap::new();
        for &(lat, lon) in &pts {
            let k = GeoIndex::key_of(&p(lat, lon)).unwrap();
            assert!(
                seen.insert(k, (lat, lon)).is_none(),
                "({lat}, {lon}) collided with {:?}",
                seen[&k]
            );
        }
        // Both zeros are the same coordinate and must land in the same cell.
        assert_eq!(
            GeoIndex::key_of(&p(0.0, 0.0)),
            GeoIndex::key_of(&p(-0.0, -0.0))
        );
        // The grid is uniform: 0.001° is the same number of steps at the origin as at 51°N. On a
        // float-bit key those two differ by ~10^7 steps, which is exactly why it could not index.
        let steps = |a: f64, b: f64| {
            i64::from(quantise(b, LAT_LO, LAT_SPAN)) - i64::from(quantise(a, LAT_LO, LAT_SPAN))
        };
        assert!(
            (steps(0.0, 0.001) - steps(51.5, 51.501)).abs() <= 1,
            "a uniform grid must not spend its resolution near zero: {} vs {}",
            steps(0.0, 0.001),
            steps(51.5, 51.501)
        );

        assert_eq!(GeoIndex::key_of(&p(f32::NAN, 0.0)), None);
        assert_eq!(GeoIndex::key_of(&Value::Int(1)), None);
    }

    /// The property the uniform grid exists for, pinned on the exact corpus that exposed the
    /// float-bit key: 100 points on the diagonal from (0,0), a 10 km search from the origin.
    ///
    /// The lossless key returned all 100 as candidates because the encoded box straddling zero
    /// spanned half the axis. A cover that cannot narrow is a full scan with extra steps, so this
    /// asserts selectivity, not just correctness.
    #[test]
    fn a_search_near_the_origin_narrows_to_its_neighbourhood() {
        let mut idx = GeoIndex::new();
        for i in 0..100u64 {
            let c = i as f32 / 100.0;
            idx.add(&p(c, c), i);
        }
        let centre = Point::new(0.0, 0.0);
        let truth: Vec<u64> = (0..100u64)
            .filter(|i| {
                let c = *i as f32 / 100.0;
                Point::new(c, c).distance(&centre) < 10_000.0
            })
            .collect();
        let candidates = ids(idx.within(&centre, 10_000.0));

        for t in &truth {
            assert!(
                candidates.contains(t),
                "id {t} is inside the circle but was missed"
            );
        }
        assert!(
            candidates.len() <= truth.len() * 3,
            "{} candidates for {} hits — the cover is not narrowing",
            candidates.len(),
            truth.len()
        );
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
                let (elat, elon) = (
                    quantise(lat, LAT_LO, LAT_SPAN),
                    quantise(lon, LON_LO, LON_SPAN),
                );
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

    /// Equality answers the grid cell, which is a superset of the points equal to the operand.
    /// Places metres apart still separate; places millimetres apart share a cell and are told
    /// apart by the retained filter, never by this.
    #[test]
    fn point_equality_answers_the_grid_cell() {
        let mut idx = GeoIndex::new();
        idx.add(&p(1.0, 2.0), 1);
        idx.add(&p(1.0, 2.000_1), 2); // ~11 m east: a different cell
        idx.add(&p(1.0, 2.000_000_01), 3); // ~1 mm east: the SAME cell as id 1
        assert_eq!(ids(idx.point(&p(1.0, 2.0))), vec![1, 3]);
        assert_eq!(ids(idx.point(&p(1.0, 2.000_1))), vec![2]);
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
