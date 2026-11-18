//! A graph's schema: its label, relationship-type and property-key names (#3757).
//!
//! Each list is a [`NameMap`]: the names in id order (position is the id), the map from
//! a name back to its id, and the list's schema-hash chain ([`super::schema_hash`]).
//! [`Schema`] holds the three and computes the hash over them.
//!
//! The lists are told apart by type: [`Labels`], [`RelationshipTypes`] and
//! [`PropertyKeys`] each implement [`SchemaList`], which gives the list's chain tag and
//! id type. A `NameMap<Labels>` hands out [`LabelId`]s, and its chain can't be passed
//! where the relationship types' is expected.

use std::{
    borrow::Borrow,
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::Index,
    sync::Arc,
};

use rustc_hash::FxHashMap;

use super::{
    attribute_store::{ATTRIBUTE_ID_NONE, AttrNameMap},
    graph::{LabelId, TypeId},
    schema_hash::{NameChain, schema_hash},
};

/// One of a schema's three name lists.
///
/// The supertraits let [`NameChain`] and [`NameMap`] derive theirs: a derive asks the
/// same of `L`.
pub trait SchemaList: Copy + Default + Eq + Debug {
    /// The byte the list's chain hashes before each name; see [`super::schema_hash`].
    const TAG: u8;
    /// What a name in this list is known by.
    type Id: Copy;
    /// The id of the name at `position`, or `None` if this list's ids can't reach it.
    fn id(position: usize) -> Option<Self::Id>;
}

/// The node labels.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Labels;

impl SchemaList for Labels {
    const TAG: u8 = 0;
    type Id = LabelId;

    fn id(position: usize) -> Option<LabelId> {
        Some(LabelId(position))
    }
}

/// The relationship types.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct RelationshipTypes;

impl SchemaList for RelationshipTypes {
    const TAG: u8 = 1;
    type Id = TypeId;

    fn id(position: usize) -> Option<TypeId> {
        Some(TypeId(position))
    }
}

/// The property keys, one list for nodes and relationships alike.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct PropertyKeys;

impl SchemaList for PropertyKeys {
    const TAG: u8 = 2;
    type Id = u16;

    /// A `u16` other than [`ATTRIBUTE_ID_NONE`], so the list is full at
    /// `MAX_ATTRIBUTES` names; [`AttrNameMap::get_or_create`] says why.
    fn id(position: usize) -> Option<u16> {
        u16::try_from(position)
            .ok()
            .filter(|&id| id != ATTRIBUTE_ID_NONE)
    }
}

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

/// One list of names in id order, with the map from a name back to its id and the
/// list's schema-hash chain.
///
/// [`Self::get_or_insert`] is the only way to add a name, and it updates all three, so
/// they can't drift apart.
#[derive(Clone)]
pub struct NameMap<L: SchemaList> {
    names: Vec<Arc<String>>,
    /// The reverse of `names`. A label test runs per row (`n:Person` for every row of
    /// a scan), so finding a name must not walk the list: with 500 labels registered,
    /// `MATCH (n) WHERE n:Person RETURN count(n)` over 10k nodes measured 2.1x the same
    /// query on a one-label schema, all of it in the walk.
    ids: FxHashMap<Key, L::Id>,
    chain: NameChain<L>,
}

/// By hand: a derive would also ask `L::Id: Default`, which an empty map doesn't need.
impl<L: SchemaList> Default for NameMap<L> {
    fn default() -> Self {
        Self {
            names: Vec::new(),
            ids: FxHashMap::default(),
            chain: NameChain::default(),
        }
    }
}

impl<L: SchemaList> NameMap<L> {
    /// The map over `names`, in id order, as an RDB load reads them.
    ///
    /// Only a malformed RDB lists a name twice. Both positions are kept, so ids still
    /// line up with the saved matrices, and the name is found at its first, as C's
    /// `GraphContext_GetSchema` finds it. The chain covers every position. Names past
    /// the last id the list has are dropped, as [`Self::get_or_insert`] would refuse
    /// them.
    #[must_use]
    pub fn from_names(mut names: Vec<Arc<String>>) -> Self {
        let mut ids = FxHashMap::default();
        ids.reserve(names.len());
        let mut end = names.len();
        for (position, name) in names.iter().enumerate() {
            let Some(id) = L::id(position) else {
                end = position;
                break;
            };
            ids.entry(Key(Arc::clone(name))).or_insert(id);
        }
        names.truncate(end);
        let chain = NameChain::of(names.iter().map(|name| name.as_str()));
        Self { names, ids, chain }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.names.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The names in id order: position is the id.
    #[must_use]
    pub fn names(&self) -> &[Arc<String>] {
        &self.names
    }

    /// The name at `position`.
    #[must_use]
    pub fn get(
        &self,
        position: usize,
    ) -> Option<&Arc<String>> {
        self.names.get(position)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Arc<String>> {
        self.names.iter()
    }

    /// The id of `name`, if the list has it.
    #[must_use]
    pub fn id_of(
        &self,
        name: &str,
    ) -> Option<L::Id> {
        self.ids.get(name).copied()
    }

    /// The id of `name`, appending it first if the list hasn't got it. `None` if the
    /// list is full, which only the property keys can be.
    pub fn get_or_insert(
        &mut self,
        name: &Arc<String>,
    ) -> Option<L::Id> {
        if let Some(id) = self.id_of(name) {
            return Some(id);
        }
        let id = L::id(self.names.len())?;
        self.chain.push(name);
        self.names.push(Arc::clone(name));
        self.ids.insert(Key(Arc::clone(name)), id);
        Some(id)
    }

    /// The schema-hash chain over the names, in id order.
    #[must_use]
    pub const fn chain(&self) -> NameChain<L> {
        self.chain
    }
}

impl<L: SchemaList> Index<usize> for NameMap<L> {
    type Output = Arc<String>;

    fn index(
        &self,
        position: usize,
    ) -> &Arc<String> {
        &self.names[position]
    }
}

impl<'a, L: SchemaList> IntoIterator for &'a NameMap<L> {
    type Item = &'a Arc<String>;
    type IntoIter = std::slice::Iter<'a, Arc<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A graph's names: its labels, relationship types and property keys, each in id order.
#[derive(Clone, Default)]
pub struct Schema {
    pub labels: NameMap<Labels>,
    pub relationship_types: NameMap<RelationshipTypes>,
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
    use crate::graph::attribute_store::MAX_ATTRIBUTES;

    fn names(list: &[&str]) -> Vec<Arc<String>> {
        list.iter()
            .map(|name| Arc::new((*name).to_string()))
            .collect()
    }

    #[test]
    fn get_or_insert_numbers_by_position_and_extends_the_chain_once() {
        let mut labels = NameMap::<Labels>::default();
        let person = Arc::new("Person".to_string());
        let city = Arc::new("City".to_string());
        assert_eq!(labels.get_or_insert(&person), Some(LabelId(0)));
        assert_eq!(labels.get_or_insert(&city), Some(LabelId(1)));
        let chain = labels.chain();
        // A name already in the list adds nothing, to the names or to the chain.
        assert_eq!(labels.get_or_insert(&person), Some(LabelId(0)));
        assert_eq!(labels.len(), 2);
        assert_eq!(labels.chain(), chain);
        assert_eq!(labels.id_of("City"), Some(LabelId(1)));
        assert_eq!(labels.id_of("Nowhere"), None);
        assert_eq!(labels.names(), names(&["Person", "City"]));
        // What an RDB load of the same list builds.
        assert_eq!(
            NameMap::<Labels>::from_names(names(&["Person", "City"])).chain(),
            chain
        );
    }

    /// A list read from an RDB keeps every position, so ids still match the saved
    /// matrices, and a name listed twice is found at its first, as C finds it.
    #[test]
    fn from_names_keeps_every_position_and_finds_the_first() {
        let types = NameMap::<RelationshipTypes>::from_names(names(&["R", "S", "R"]));
        assert_eq!(types.len(), 3);
        assert_eq!(types[2].as_str(), "R");
        assert_eq!(types.id_of("R"), Some(TypeId(0)));
        assert_eq!(types.id_of("S"), Some(TypeId(1)));
        assert_eq!(types.chain(), NameChain::of(["R", "S", "R"]));
    }

    /// A list longer than its ids reach keeps only the names it can number, so every
    /// name has an id and the chain matches what `get_or_insert` would have built.
    #[test]
    fn from_names_drops_names_past_the_last_id() {
        let all: Vec<Arc<String>> = (0..=MAX_ATTRIBUTES)
            .map(|i| Arc::new(format!("k{i}")))
            .collect();
        let keys = AttrNameMap::from_names(all.clone());
        assert_eq!(keys.len(), MAX_ATTRIBUTES);
        assert_eq!(keys.get(MAX_ATTRIBUTES), None);
        assert_eq!(keys.id_of(&format!("k{MAX_ATTRIBUTES}")), None);
        let mut inserted = AttrNameMap::default();
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
