//! Deterministic PRNG for reproducible vertex-order permutations
//! and random tie-breaking. SplitMix64 (Steele, Lea & Flood 2014)
//! is small, fast, and has enough quality for shuffling vertex
//! orders during matching and initial bisection. FERAL does not
//! match METIS byte-for-byte by design, so a cryptographic PRNG
//! would be overkill.

#[derive(Debug, Clone)]
pub struct SplitMix {
    state: u64,
}

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)`. `n == 0` is undefined; callers
    /// must guard. Uses rejection sampling to avoid modulo bias for
    /// small `n`.
    pub fn gen_range(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next_u64();
            if r < zone {
                return r % n;
            }
        }
    }

    /// Fisher-Yates shuffle in place. Deterministic given the seed.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.gen_range((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut a = SplitMix::new(42);
        let mut b = SplitMix::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SplitMix::new(1);
        let mut b = SplitMix::new(2);
        let x: u64 = (0..10).fold(0u64, |acc, _| acc.wrapping_add(a.next_u64()));
        let y: u64 = (0..10).fold(0u64, |acc, _| acc.wrapping_add(b.next_u64()));
        assert_ne!(x, y);
    }

    #[test]
    fn shuffle_is_permutation() {
        let mut v: Vec<feral_ordering_core::Idx> = (0..100).collect();
        let mut rng = SplitMix::new(7);
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn gen_range_bounds() {
        let mut rng = SplitMix::new(99);
        for _ in 0..1000 {
            let r = rng.gen_range(7);
            assert!(r < 7);
        }
    }
}
