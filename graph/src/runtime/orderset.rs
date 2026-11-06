//! Insertion-ordered set preserving element order.
//!
//! This module provides [`OrderSet`], a set that maintains insertion order.
//! Used for:
//!
//! - Label collections on nodes (consistent ordering)
//! - Property name lists (deterministic iteration)
//! - Any context where set order must be predictable
//!
//! ## Implementation
//!
//! Uses a `Vec<T>` internally with O(n) membership check. Efficient for
//! small sets while guaranteeing order.

use std::ops::Index;

/// A set that preserves insertion order during iteration.
///
/// Elements are compared by equality. Duplicate insertions replace the
/// existing element (returns the old value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderSet<T> {
    vec: Vec<T>,
}

impl<T> Default for OrderSet<T> {
    fn default() -> Self {
        Self { vec: Vec::new() }
    }
}

impl<T: PartialEq> OrderSet<T> {
    /// Wrap `vec` as a set, keeping the first occurrence of each element.
    ///
    /// Callers normally pass distinct elements (the only one, label
    /// reordering in the planner, permutes an existing set). The duplicate
    /// check still runs in release builds because a duplicate would break the
    /// set: `remove` drops only the first copy, so the element would stay
    /// present. It is in place and allocation-free, O(n^2) in the element
    /// count, which is small here (a node's labels) and off the per-row path.
    #[must_use]
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let mut i = 1;
        while i < vec.len() {
            if vec[..i].contains(&vec[i]) {
                vec.remove(i);
            } else {
                i += 1;
            }
        }
        Self { vec }
    }

    pub fn insert(
        &mut self,
        value: T,
    ) -> Option<T> {
        for v in &mut self.vec {
            if *v == value {
                let old = std::mem::replace(v, value);
                return Some(old);
            }
        }
        self.vec.push(value);
        None
    }

    pub fn remove(
        &mut self,
        value: &T,
    ) {
        if let Some(pos) = self.vec.iter().position(|v| v == value) {
            self.vec.remove(pos);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.vec.iter()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.vec.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn contains(
        &self,
        value: &T,
    ) -> bool {
        for v in &self.vec {
            if v == value {
                return true;
            }
        }
        false
    }

    pub fn extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) {
        for value in iter {
            self.insert(value);
        }
    }

    pub fn clear(&mut self) {
        self.vec.clear();
    }

    pub fn get_index_of(
        &self,
        value: &T,
    ) -> Option<usize> {
        for (i, v) in self.vec.iter().enumerate() {
            if v == value {
                return Some(i);
            }
        }
        None
    }

    #[must_use]
    pub fn get(
        &self,
        index: usize,
    ) -> Option<&T> {
        self.vec.get(index)
    }
}

impl<T: PartialEq> FromIterator<T> for OrderSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::default();
        set.extend(iter);
        set
    }
}

impl<T: PartialEq> Index<usize> for OrderSet<T> {
    type Output = T;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        self.vec.get(index).expect("no entry found for key")
    }
}

impl<T> IntoIterator for OrderSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_keeps_distinct_elements() {
        let mut s = OrderSet::from_vec(vec![1, 2, 3]);
        s.remove(&2);
        assert!(!s.contains(&2));
        assert_eq!(s.len(), 2);
    }

    /// A duplicate passed to `from_vec` is dropped in every build, so
    /// `remove` really removes the element.
    #[test]
    fn from_vec_drops_duplicates() {
        let mut s = OrderSet::from_vec(vec![1, 2, 1, 3, 2]);
        assert_eq!(s.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
        s.remove(&1);
        assert!(!s.contains(&1));
        assert_eq!(s.len(), 2);
    }
}
