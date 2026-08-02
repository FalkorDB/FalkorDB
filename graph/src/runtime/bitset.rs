//! Compact bit set with inline storage optimized for small variable counts.
//!
//! Most Cypher queries bind fewer than 64 variables, so [`BitSet`] stores
//! the first 64 bits inline (no heap allocation). Additional words spill
//! into a [`ThinVec`] only when needed.
//!
//! ```text
//!  word 0 (inline)      word 1 (overflow[0])  word 2 (overflow[1])
//! ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
//! │ bits  0 ..  63  │  │ bits 64 .. 127  │  │ bits 128 .. 191 │  ...
//! └─────────────────┘  └─────────────────┘  └─────────────────┘
//! ```
//!
//! Supports `set`, `test`, and `union` — the three operations needed for
//! tracking bound variables in a [`Row`](super::row::Row).

use thin_vec::ThinVec;

/// A compact bit vector with inline storage for the first 64 bits.
///
/// Most queries use fewer than 64 variables, so the common case avoids
/// any heap allocation. When more than 64 bits are needed, additional
/// words spill into a `ThinVec<u64>`.
#[derive(Default, Clone)]
pub struct BitSet {
    inline: u64,
    overflow: ThinVec<u64>,
}

impl BitSet {
    pub fn set(
        &mut self,
        bit: usize,
    ) {
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);
        if word == 0 {
            self.inline |= mask;
        } else {
            let idx = word - 1;
            if idx >= self.overflow.len() {
                self.overflow.resize(idx + 1, 0);
            }
            self.overflow[idx] |= mask;
        }
    }

    #[must_use]
    pub fn test(
        &self,
        bit: usize,
    ) -> bool {
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);
        if word == 0 {
            self.inline & mask != 0
        } else {
            let idx = word - 1;
            idx < self.overflow.len() && self.overflow[idx] & mask != 0
        }
    }

    pub fn union(
        &mut self,
        other: &Self,
    ) {
        self.inline |= other.inline;
        if other.overflow.len() > self.overflow.len() {
            self.overflow.resize(other.overflow.len(), 0);
        }
        for (a, b) in self.overflow.iter_mut().zip(other.overflow.iter()) {
            *a |= b;
        }
    }

    pub fn clear(
        &mut self,
        bit: usize,
    ) {
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);
        if word == 0 {
            self.inline &= !mask;
        } else {
            let idx = word - 1;
            if idx < self.overflow.len() {
                self.overflow[idx] &= !mask;
            }
        }
    }

    /// Returns true if no bits are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inline == 0 && self.overflow.iter().all(|&w| w == 0)
    }
}
