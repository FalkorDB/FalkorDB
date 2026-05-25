//! Variable-binding environment for query execution.
//!
//! An [`Env`] is a tuple of variable bindings that flows through the
//! pull-based operator pipeline. Each variable slot is indexed by
//! `Variable.id` (assigned during binding), and a [`BitSet`] tracks
//! which slots have been explicitly bound (vs padding `Null`s).
//!
//! ```text
//!  Env (one row flowing through the plan)
//! ┌────────┬──────────────────────────────────────────┐
//! │ bound  │  0  1  1  0  1    (BitSet)               │
//! ├────────┼──────────────────────────────────────────┤
//! │ values │ Null  Node(3)  Int(42)  Null  "Alice"    │
//! └────────┴──────────────────────────────────────────┘
//!            slot 0  slot 1  slot 2  slot 3  slot 4
//! ```

use std::hash::Hash;

use crate::parser::ast::Variable;
use crate::runtime::bitset::BitSet;
use crate::runtime::pool::{Pool, Pooled};
use crate::runtime::value::Value;

pub struct Env<'a> {
    values: Pooled<'a, Value>,
    bound: BitSet,
    /// Tracks which input row this env originated from in batched correlated
    /// sub-plans (Apply, SemiApply, Optional, etc.). Set by the parent
    /// operator before passing to the sub-plan; preserved through clone_pooled.
    pub origin_row: u32,
}

impl<'a> Env<'a> {
    /// Create a new empty Env backed by the given pool.
    #[must_use]
    pub fn new(pool: &'a Pool<Value>) -> Self {
        Self {
            values: pool.acquire(0),
            bound: BitSet::default(),
            origin_row: 0,
        }
    }

    /// Create a new empty Env with the given capacity, backed by the pool.
    #[must_use]
    pub fn with_capacity(
        num_vars: usize,
        pool: &'a Pool<Value>,
    ) -> Self {
        Self {
            values: pool.acquire(num_vars),
            bound: BitSet::default(),
            origin_row: 0,
        }
    }

    pub fn insert(
        &mut self,
        key: &Variable,
        value: Value,
    ) {
        self.insert_by_id(key.id, value);
    }

    /// Inserts a value at the given variable id slot.
    /// This is used by the batch system to reconstruct envs from columns.
    pub fn insert_by_id(
        &mut self,
        id: u32,
        value: Value,
    ) {
        if self.values.len() <= id as usize {
            self.values.resize_with(id as usize + 1, || Value::Null);
        }
        self.values[id as usize] = value;
        self.bound.set(id as usize);
    }

    /// Returns true if the variable was explicitly inserted (even if set to Null).
    /// Returns false for padding Null slots that were never explicitly set.
    #[must_use]
    pub fn is_bound(
        &self,
        key: &Variable,
    ) -> bool {
        self.bound.test(key.id as usize)
    }

    /// Clear the bound bit for a variable slot, making it appear unbound.
    ///
    /// Note: this intentionally does NOT reset the stored value. The bound bit
    /// is only checked by `Hash` / `merge` / `is_bound`; readers like `get()`
    /// still see the old value, but that is acceptable because `unbind` is only
    /// called at the end of aggregate finalization where the accumulator ID
    /// may alias a different output variable's slot — clearing the value would
    /// destroy that result.
    pub fn unbind(
        &mut self,
        key: &Variable,
    ) {
        self.bound.clear(key.id as usize);
    }

    #[must_use]
    pub fn get(
        &self,
        key: &Variable,
    ) -> Option<&Value> {
        self.values.get(key.id as usize)
    }

    /// Returns a reference to the value at the given variable id slot.
    #[must_use]
    pub fn get_by_id(
        &self,
        id: u32,
    ) -> Option<&Value> {
        self.values.get(id as usize)
    }

    /// Takes ownership of a value from the environment, replacing it with `Null`.
    ///
    /// This method is designed for aggregation optimizations where we need to transfer
    /// ownership of large accumulated values (like lists with millions of items) without
    /// cloning them. By replacing the environment entry with `Null`, we ensure the value
    /// can be moved (not cloned) to the aggregation function.
    ///
    /// # Returns
    /// - `Some(value)` if the key exists and contains a non-Null value
    /// - `None` if the key doesn't exist or already contains `Null`
    ///
    /// # Usage
    /// Prefer this over `get()` when:
    /// - You need exclusive ownership of a value
    /// - The value is expensive to clone (e.g., large collections)
    /// - The environment slot won't be read again before being overwritten
    pub fn take(
        &mut self,
        key: &Variable,
    ) -> Option<Value> {
        self.values.get_mut(key.id as usize).and_then(|value| {
            match std::mem::replace(value, Value::Null) {
                Value::Null => None,
                v => Some(v),
            }
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Merge values from another Env by reference, without requiring
    /// the caller to clone and move a temporary Env.
    pub fn merge(
        &mut self,
        other: &Self,
    ) {
        if self.values.len() < other.values.len() {
            self.values.resize_with(other.values.len(), || Value::Null);
        }
        for (key, value) in other.values.iter().enumerate() {
            if !other.bound.test(key) {
                continue;
            }
            self.values[key] = value.clone();
            self.bound.set(key);
        }
        self.bound.union(&other.bound);
    }

    /// Clone this Env into a new one backed by the given pool.
    #[must_use]
    pub fn clone_pooled<'b>(
        &self,
        pool: &'b Pool<Value>,
    ) -> Env<'b> {
        let mut new_values = pool.acquire(self.values.len());
        new_values.extend_from_slice(&self.values);
        Env {
            values: new_values,
            bound: self.bound.clone(),
            origin_row: self.origin_row,
        }
    }

    /// Clone values and bound set into an unmanaged (non-pooled) snapshot.
    #[must_use]
    pub fn to_raw(&self) -> (Vec<Value>, BitSet) {
        (self.values.to_vec(), self.bound.clone())
    }
}

impl AsRef<Vec<Value>> for Env<'_> {
    fn as_ref(&self) -> &Vec<Value> {
        &self.values
    }
}

impl Hash for Env<'_> {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        for (key, value) in self.values.iter().enumerate() {
            if matches!(value, Value::Null) && !self.bound.test(key) {
                continue;
            }
            key.hash(state);
            value.hash(state);
        }
    }
}
