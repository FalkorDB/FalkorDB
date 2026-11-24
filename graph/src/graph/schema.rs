//! A graph's schema: its label, relationship-type and property-key names (#3757).
//!
//! Each list is a [`NameMap`]: the names in id order (position is the id), the map from
//! a name back to its id, and the list's schema-hash chain ([`super::schema_hash`]).
//! [`Schema`] holds the three and computes the hash over them.

use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    ops::Index,
    sync::Arc,
};

use rustc_hash::FxHashMap;

use super::{
    attribute_store::{AttrNameMap, MAX_ATTRIBUTES},
    schema_hash::{chain_of, extend_chain, schema_hash},
};

/// A name as a map key, looked up by `&str` so a lookup needn't build an `Arc`.
#[derive(Clone, PartialEq, Eq)]
struct Key(Arc<String>);

/// Hashes the `str` it borrows as, which `Borrow` requires.
impl Hash for Key {
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        self.0.as_str().hash(state);
    }
}

impl Borrow<str> for Key {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// One list's names in id order, with the map from a name back to its id and the
/// list's schema-hash chain. Only [`NameMap`] changes it, so the three can't drift
/// apart.
#[derive(Clone, Default)]
pub struct NameList {
    names: Vec<Arc<String>>,
    /// The reverse of `names`. A label test runs per row (`n:Person` for every row of
    /// a scan), so finding a name must not walk the list: with 500 labels registered,
    /// `MATCH (n) WHERE n:Person RETURN count(n)` over 10k nodes measured 2.1x the same
    /// query on a one-label schema, all of it in the walk.
    ids: FxHashMap<Key, usize>,
    chain: u64,
}

/// One of a schema's three name lists. The variant decides the list's tag in the
/// schema hash and how many names it can hold; an empty list is, for example,
/// `NameMap::Labels(NameList::default())`.
///
/// [`Self::get_or_insert`] is the only way to add a name.
#[derive(Clone)]
pub enum NameMap {
    Labels(NameList),
    RelationshipTypes(NameList),
    PropertyKeys(NameList),
}

impl NameMap {
    /// The list `list` makes, holding `names` in id order, as an RDB load reads them:
    /// `NameMap::from_names(NameMap::Labels, names)`.
    ///
    /// Only a malformed RDB lists a name twice. Both positions are kept, so ids still
    /// line up with the saved matrices, and the name is found at its first, as C's
    /// `GraphContext_GetSchema` finds it. The chain covers every position. Names past
    /// what the list can hold are dropped, as [`Self::get_or_insert`] would refuse them.
    #[must_use]
    pub fn from_names(
        list: fn(NameList) -> Self,
        mut names: Vec<Arc<String>>,
    ) -> Self {
        let mut map = list(NameList::default());
        names.truncate(map.capacity());
        let tag = map.tag();
        let inner = map.inner_mut();
        inner.ids.reserve(names.len());
        for (id, name) in names.iter().enumerate() {
            inner.ids.entry(Key(Arc::clone(name))).or_insert(id);
        }
        inner.chain = chain_of(tag, names.iter().map(|name| name.as_str()));
        inner.names = names;
        map
    }

    /// The list's tag in the schema hash.
    const fn tag(&self) -> u8 {
        match self {
            Self::Labels(_) => 0,
            Self::RelationshipTypes(_) => 1,
            Self::PropertyKeys(_) => 2,
        }
    }

    /// How many names the list can hold. Property-key ids are `u16`s other than
    /// `ATTRIBUTE_ID_NONE` ([`AttrNameMap::get_or_create`] says why); the other lists
    /// have no limit.
    const fn capacity(&self) -> usize {
        match self {
            Self::PropertyKeys(_) => MAX_ATTRIBUTES,
            Self::Labels(_) | Self::RelationshipTypes(_) => usize::MAX,
        }
    }

    const fn inner(&self) -> &NameList {
        match self {
            Self::Labels(names) | Self::RelationshipTypes(names) | Self::PropertyKeys(names) => {
                names
            }
        }
    }

    const fn inner_mut(&mut self) -> &mut NameList {
        match self {
            Self::Labels(names) | Self::RelationshipTypes(names) | Self::PropertyKeys(names) => {
                names
            }
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner().names.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.inner().names.is_empty()
    }

    /// The names in id order: position is the id.
    #[must_use]
    pub fn names(&self) -> &[Arc<String>] {
        &self.inner().names
    }

    /// The name with id `id`.
    #[must_use]
    pub fn get(
        &self,
        id: usize,
    ) -> Option<&Arc<String>> {
        self.inner().names.get(id)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Arc<String>> {
        self.inner().names.iter()
    }

    /// The id of `name`, if the list has it.
    #[must_use]
    pub fn id_of(
        &self,
        name: &str,
    ) -> Option<usize> {
        self.inner().ids.get(name).copied()
    }

    /// The id of `name`, appending it first if the list hasn't got it. `None` if the
    /// list is full, which only the property keys can be.
    pub fn get_or_insert(
        &mut self,
        name: &Arc<String>,
    ) -> Option<usize> {
        if let Some(id) = self.id_of(name) {
            return Some(id);
        }
        if self.len() >= self.capacity() {
            return None;
        }
        let tag = self.tag();
        let inner = self.inner_mut();
        let id = inner.names.len();
        inner.chain = extend_chain(tag, inner.chain, name);
        inner.names.push(Arc::clone(name));
        inner.ids.insert(Key(Arc::clone(name)), id);
        Some(id)
    }

    /// The schema-hash chain over the names, in id order.
    #[must_use]
    pub const fn chain(&self) -> u64 {
        self.inner().chain
    }
}

impl Index<usize> for NameMap {
    type Output = Arc<String>;

    fn index(
        &self,
        id: usize,
    ) -> &Arc<String> {
        &self.inner().names[id]
    }
}

impl<'a> IntoIterator for &'a NameMap {
    type Item = &'a Arc<String>;
    type IntoIter = std::slice::Iter<'a, Arc<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A graph's names: its labels, relationship types and property keys, each in id order.
#[derive(Clone)]
pub struct Schema {
    pub labels: NameMap,
    pub relationship_types: NameMap,
    /// One list for the whole graph, shared by the node and relationship stores, so an
    /// id names the same property wherever it appears: in a span, the RDB, a
    /// `GRAPH.EFFECT`, or a compact reply.
    ///
    /// Deliberately *not* per store. Two dictionaries numbered the same name
    /// differently for nodes and relationships, and because effects put a bare id on
    /// the wire, an RDB-seeded replica resolved it against different numbering and wrote
    /// to the wrong attribute (#2457). C has the same shape: one `attributes` array on
    /// `GraphContext`, two `DataBlock`s.
    pub property_keys: AttrNameMap,
}

impl Default for Schema {
    fn default() -> Self {
        Self {
            labels: NameMap::Labels(NameList::default()),
            relationship_types: NameMap::RelationshipTypes(NameList::default()),
            property_keys: NameMap::PropertyKeys(NameList::default()),
        }
    }
}

impl Schema {
    /// The schema hash: one value standing for the three lists, in id order. See
    /// [`super::schema_hash`].
    #[must_use]
    pub const fn hash(&self) -> u64 {
        schema_hash(
            self.labels.chain(),
            self.relationship_types.chain(),
            self.property_keys.chain(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<Arc<String>> {
        list.iter()
            .map(|name| Arc::new((*name).to_string()))
            .collect()
    }

    #[test]
    fn get_or_insert_numbers_by_position_and_extends_the_chain_once() {
        let mut labels = NameMap::Labels(NameList::default());
        let person = Arc::new("Person".to_string());
        let city = Arc::new("City".to_string());
        assert_eq!(labels.get_or_insert(&person), Some(0));
        assert_eq!(labels.get_or_insert(&city), Some(1));
        let chain = labels.chain();
        // A name already in the list adds nothing, to the names or to the chain.
        assert_eq!(labels.get_or_insert(&person), Some(0));
        assert_eq!(labels.len(), 2);
        assert_eq!(labels.chain(), chain);
        assert_eq!(labels.id_of("City"), Some(1));
        assert_eq!(labels.id_of("Nowhere"), None);
        assert_eq!(labels.names(), names(&["Person", "City"]));
        // What an RDB load of the same list builds.
        assert_eq!(
            NameMap::from_names(NameMap::Labels, names(&["Person", "City"])).chain(),
            chain
        );
    }

    /// A list read from an RDB keeps every position, so ids still match the saved
    /// matrices, and a name listed twice is found at its first, as C finds it.
    #[test]
    fn from_names_keeps_every_position_and_finds_the_first() {
        let types = NameMap::from_names(NameMap::RelationshipTypes, names(&["R", "S", "R"]));
        assert_eq!(types.len(), 3);
        assert_eq!(types[2].as_str(), "R");
        assert_eq!(types.id_of("R"), Some(0));
        assert_eq!(types.id_of("S"), Some(1));
        assert_eq!(types.chain(), chain_of(1, ["R", "S", "R"]));
    }

    /// A list longer than it can hold keeps only what fits, so every name has an id and
    /// the chain matches what `get_or_insert` would have built.
    #[test]
    fn from_names_drops_names_past_the_capacity() {
        let all: Vec<Arc<String>> = (0..=MAX_ATTRIBUTES)
            .map(|i| Arc::new(format!("k{i}")))
            .collect();
        let keys = NameMap::from_names(NameMap::PropertyKeys, all.clone());
        assert_eq!(keys.len(), MAX_ATTRIBUTES);
        assert_eq!(keys.get(MAX_ATTRIBUTES), None);
        assert_eq!(keys.id_of(&format!("k{MAX_ATTRIBUTES}")), None);
        let mut inserted = NameMap::PropertyKeys(NameList::default());
        for name in &all {
            inserted.insert(name);
        }
        assert_eq!(keys.chain(), inserted.chain());
    }

    #[test]
    fn schema_hash_covers_the_three_lists() {
        let mut schema = Schema::default();
        assert_eq!(schema.hash(), 0);
        for name in ["Person", "City"] {
            schema.labels.get_or_insert(&Arc::new(name.to_string()));
        }
        schema
            .relationship_types
            .get_or_insert(&Arc::new("LIVES_IN".to_string()));
        for name in ["name", "age"] {
            schema
                .property_keys
                .get_or_create(&Arc::new(name.to_string()));
        }
        // The value `schema_hash::tests::golden_values` pins for these lists.
        assert_eq!(schema.hash(), 17_300_287_264_737_141_393);
    }
}
