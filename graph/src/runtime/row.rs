//! Row abstraction for expression evaluation.
//!
//! The expression evaluator ([`ExprEval`](super::eval::ExprEval)) reads a
//! single row of variable bindings through the [`RowView`] trait. This lets a
//! single evaluator body run against several concrete row sources:
//!
//! - [`Row`] — an owned, pool-free binding tuple used by buffering/correlated
//!   operators and by scoped sub-evaluations (list comprehension, `reduce`,
//!   quantifiers) that need to extend a row with an extra binding.
//! - `BatchRow` (in [`batch`](super::batch)) — a borrowed view of one row of a
//!   columnar [`Batch`](super::batch::Batch), reading directly from columns.
//!
//! ```text
//!  RowView::value_at(var_id) -> Option<Value>
//!     │
//!     ├── Row       → owned Vec<Value> slot
//!     └── BatchRow  → Column::get(row) (typed column → Value)
//! ```

use crate::parser::ast::Variable;
use crate::runtime::bitset::BitSet;
use crate::runtime::value::Value;
use std::hash::Hash;

/// Read access to a single row of variable bindings.
///
/// `var_id` is a `Variable.id` (assigned during binding). Returns `None` when
/// the slot is out of range (variable never bound in this row); returns
/// `Some(Value::Null)` for an in-range slot that holds `Null`.
pub trait RowView {
    /// Returns a clone of the value bound to `var_id`, or `None` if the slot
    /// is out of range (unbound).
    fn value_at(
        &self,
        var_id: u32,
    ) -> Option<Value>;

    /// Snapshots this row into an owned [`Row`]. Used by scoped
    /// sub-evaluations (list comprehension, `reduce`, quantifiers) that extend
    /// the row with an extra loop/accumulator binding without mutating the
    /// original.
    fn to_owned_row(&self) -> Row;
}

/// An owned, pool-free tuple of variable bindings.
///
/// `Row` is the columnar runtime's row-oriented binding tuple: a dense
/// `Vec<Value>` indexed by `Variable.id`, plus a
/// [`BitSet`] recording which slots were explicitly bound.
#[derive(Clone, Default)]
pub struct Row {
    values: Vec<Value>,
    bound: BitSet,
    /// Origin row index, propagated through correlated sub-plans (Optional,
    /// Apply) to map result rows back to their input row.
    pub origin_row: u32,
}

impl Row {
    /// Creates an empty row with no bindings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty row sized to hold `num_vars` slots.
    #[must_use]
    pub fn with_capacity(num_vars: usize) -> Self {
        Self {
            values: Vec::with_capacity(num_vars),
            bound: BitSet::default(),
            origin_row: 0,
        }
    }

    /// Creates a row directly from a dense value vector and bound set.
    #[must_use]
    pub const fn from_raw(
        values: Vec<Value>,
        bound: BitSet,
    ) -> Self {
        Self {
            values,
            bound,
            origin_row: 0,
        }
    }

    /// Binds `value` to the slot for `key`.
    pub fn insert(
        &mut self,
        key: &Variable,
        value: Value,
    ) {
        self.insert_by_id(key.id, value);
    }

    /// Binds `value` to the slot for variable id `id`, growing as needed.
    pub fn insert_by_id(
        &mut self,
        id: u32,
        value: Value,
    ) {
        let idx = id as usize;
        if self.values.len() <= idx {
            self.values.resize_with(idx + 1, || Value::Null);
        }
        self.values[idx] = value;
        self.bound.set(idx);
    }

    /// Returns a reference to the value at `id`, or `None` if out of range.
    #[must_use]
    pub fn get_by_id(
        &self,
        id: u32,
    ) -> Option<&Value> {
        self.values.get(id as usize)
    }

    /// Returns a reference to the value bound to `key`, or `None` if out of
    /// range.
    #[must_use]
    pub fn get(
        &self,
        key: &Variable,
    ) -> Option<&Value> {
        self.values.get(key.id as usize)
    }

    /// Takes ownership of the value bound to `key`, replacing it with `Null`.
    /// Returns `None` if the slot is out of range or already `Null`.
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

    /// Clears the bound bit for `key`, leaving the stored value intact.
    pub fn unbind(
        &mut self,
        key: &Variable,
    ) {
        self.bound.clear(key.id as usize);
    }

    /// Number of value slots in this row (highest bound id + 1, padded).
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the row has no value slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns true if any variable slot is bound.
    #[must_use]
    pub fn has_bindings(&self) -> bool {
        !self.bound.is_empty()
    }

    /// Clones this row and binds `value` to variable id `id` in the clone.
    #[must_use]
    pub fn clone_with(
        &self,
        id: u32,
        value: Value,
    ) -> Row {
        let mut row = self.clone();
        row.insert_by_id(id, value);
        row
    }

    /// Returns true if the slot for variable id `id` has its bound bit set.
    #[must_use]
    pub fn is_bound_by_id(
        &self,
        id: u32,
    ) -> bool {
        self.bound.test(id as usize)
    }

    /// Merges the bound slots of `other` into `self`, overwriting on conflict.
    /// Only slots that are bound in `other` are copied, preserving the
    /// value-present-but-unbound distinction.
    pub fn merge(
        &mut self,
        other: &Row,
    ) {
        for id in 0..other.values.len() {
            if other.bound.test(id) {
                self.insert_by_id(id as u32, other.values[id].clone());
            }
        }
    }

    /// Clears the bound bit for variable id `id`, leaving the stored value
    /// intact, for value-present-but-unbound columnar slots.
    pub fn unbind_by_id(
        &mut self,
        id: u32,
    ) {
        self.bound.clear(id as usize);
    }
}

impl RowView for Row {
    fn value_at(
        &self,
        var_id: u32,
    ) -> Option<Value> {
        self.values.get(var_id as usize).cloned()
    }

    fn to_owned_row(&self) -> Row {
        self.clone()
    }
}

impl Hash for Row {
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
