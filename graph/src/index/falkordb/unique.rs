//! The native index's side of a unique-constraint check: which entities may already hold a
//! member's key. The check itself, comparing those candidates' keys with the members', is
//! `unique_violated` in `runtime::pending`; this only says where to look.

use std::{cell::RefCell, sync::Arc};

use itertools::Either;
use rustc_hash::FxHashSet;

use crate::{entity_type::EntityType, runtime::value::Value};

use super::falkordb_index::{EqualCandidates, FalkorDbIndex};

/// The native lookups for one member of a unique constraint: one per
/// constrained property whose column can say which entities hold that
/// property's value. Any one of them is enough, because `unique_violated`
/// re-checks each candidate's whole key against the graph, so
/// [`fewest_candidates`] reads only the smallest. `None` when no column can
/// answer for any of them.
pub(crate) fn native_unique_query<'g>(
    index: &'g FalkorDbIndex,
    entity: EntityType,
    label: &Arc<String>,
    properties: &[Arc<String>],
    values: &[Value],
) -> Option<Vec<EqualCandidates<'g>>> {
    let lookups: Vec<_> = properties
        .iter()
        .zip(values)
        .filter_map(|(attr, value)| index.equal_candidates(entity, label, attr, value))
        .collect();
    (!lookups.is_empty()).then_some(lookups)
}

/// The lookups one unique-constraint check has read. `unique_violated` compares
/// each candidate with every member's key, so a set of lookups read once has
/// been checked for every member, and a member whose lookups are the same as an
/// earlier member's needs no read. That is the case for a key with a member no
/// column answers for (a string id, today) beside one most of the label shares
/// (a tenant): every member looks up only the tenant, and reading it once per
/// member would make a bulk insert quadratic.
#[derive(Default)]
pub(crate) struct ReadOnce(RefCell<FxHashSet<Vec<(usize, u64)>>>);

impl ReadOnce {
    /// The candidates of `lookups` ([`fewest_candidates`]), or none when an
    /// earlier member had the same lookups.
    pub(crate) fn candidates(
        &self,
        lookups: Vec<EqualCandidates<'_>>,
    ) -> impl Iterator<Item = u64> + use<> {
        let mut ids: Vec<_> = lookups.iter().map(EqualCandidates::id).collect();
        ids.sort_unstable();
        let first = self.0.borrow_mut().insert(ids);
        first
            .then(|| fewest_candidates(lookups))
            .into_iter()
            .flatten()
    }
}

/// The candidates of the smallest of `lookups`, which all answer for one
/// member's key. A key's first property may be shared by most of the label (a
/// tenant), and reading its lookup for every member of a bulk insert costs a
/// scan of that share per member. Each round takes one entity from every
/// lookup, so the first to run out is the smallest, and finding it reads at
/// most `lookups.len()` times as many entities as it returns. Each lookup's
/// rows own their tree snapshot, so the result borrows nothing (`use<>`).
pub(crate) fn fewest_candidates(
    lookups: Vec<EqualCandidates<'_>>
) -> impl Iterator<Item = u64> + use<> {
    let mut rows: Vec<_> = lookups.iter().map(EqualCandidates::rows).collect();
    if rows.len() == 1 {
        return Either::Left(rows.pop().expect("one lookup"));
    }
    let mut read = vec![Vec::new(); rows.len()];
    loop {
        for (i, it) in rows.iter_mut().enumerate() {
            match it.next() {
                Some(id) => read[i].push(id),
                None => return Either::Right(read.swap_remove(i).into_iter()),
            }
        }
    }
}
