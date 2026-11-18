//! The schema hash: one 64-bit value standing for a graph's label, relationship-type and
//! property-key names, in id order (#3757).
//!
//! Clients cache those names to decode `--compact` replies, which carry ids. A graph
//! dropped and recreated under the same key reuses ids 0, 1, 2… for other names, and the
//! graph version can't tell: it can be equal across a recreate, and differs between a
//! primary and its replica. A client that sends `SCHEMA_HASH h` gets this value back
//! with the reply, and the three lists when it differs from `h`.
//!
//! ## Definition
//!
//! One chain per list, each starting at 0:
//!
//! ```text
//! h[0]   = 0
//! h[i+1] = XXH3-64(tag ‖ h[i] ‖ len(name) ‖ name)     seed 0; h[i], len as u64 LE
//! S      = h_labels + h_types + h_keys                mod 2^64
//! ```
//!
//! `tag` is one byte: 0 for labels, 1 for relationship types, 2 for property keys. An
//! empty schema is 0. Ids are positions, so order counts: labels `[X, Y]` and `[Y, X]`
//! give different values.
//!
//! `S` depends on nothing but the three lists, so a primary and its replicas agree, and
//! so do a server before and after a restart, and the old and new servers of a rolling
//! upgrade, for as long as this definition holds. `golden_values` below pins it with
//! values computed by a separate XXH3 implementation (Python's `xxhash`).
//!
//! A chain is extended where its list grows (each list has one writer) and rebuilt from
//! the list on RDB load. Both give the same value, so no hash is stored in the RDB.

use xxhash_rust::xxh3::Xxh3Default;

/// The chain over one list of names; `TAG` says which list.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct NameChain<const TAG: u8>(u64);

/// The chain over the label names.
pub type LabelChain = NameChain<0>;
/// The chain over the relationship-type names.
pub type TypeChain = NameChain<1>;
/// The chain over the property-key names.
pub type KeyChain = NameChain<2>;

impl<const TAG: u8> NameChain<TAG> {
    /// Extend the chain by `name`, the next name appended to its list.
    pub fn push(
        &mut self,
        name: &str,
    ) {
        // Streamed, so no heap buffer is built for the input: XXH3 gives the same value
        // as hashing the concatenation in one call.
        let mut hasher = Xxh3Default::new();
        hasher.update(&[TAG]);
        hasher.update(&self.0.to_le_bytes());
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        self.0 = hasher.digest();
    }

    /// The chain over a whole list: what pushing its names one by one gives.
    pub fn of<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut chain = Self::default();
        for name in names {
            chain.push(name);
        }
        chain
    }
}

/// `S`, the schema hash, from the three chains.
#[must_use]
pub const fn schema_hash(
    labels: LabelChain,
    types: TypeChain,
    keys: KeyChain,
) -> u64 {
    labels.0.wrapping_add(types.0).wrapping_add(keys.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(
        labels: &[&str],
        types: &[&str],
        keys: &[&str],
    ) -> u64 {
        schema_hash(
            LabelChain::of(labels.iter().copied()),
            TypeChain::of(types.iter().copied()),
            KeyChain::of(keys.iter().copied()),
        )
    }

    /// Replicas, restarts and rolling upgrades only agree if every build computes the
    /// same value. These were computed with Python's `xxhash.xxh3_64_intdigest` over the
    /// byte layout in the module docs, not with this code; a change to the layout, the
    /// tags, the seed or the hash function fails here.
    #[test]
    fn golden_values() {
        assert_eq!(hash(&[], &[], &[]), 0);
        assert_eq!(LabelChain::of(["X"]).0, 0xb12c_4238_bffd_bebf);
        assert_eq!(TypeChain::of(["X"]).0, 0x6648_e68c_fdf9_384f);
        assert_eq!(KeyChain::of(["X"]).0, 0xb04e_4cf3_ee86_6a17);
        assert_eq!(hash(&["Person"], &[], &["name"]), 145_606_067_124_355_331);
        // Above i64::MAX: the reply sends it as the negative i64 with the same bits.
        assert_eq!(
            hash(&["Person", "City"], &["LIVES_IN"], &["name", "age"]),
            17_300_287_264_737_141_393
        );
        // The length prefix carries an interior NUL rather than ending the name there.
        assert_eq!(hash(&["a\0b"], &[], &[]), 17_423_520_930_716_391_286);
    }

    #[test]
    fn order_and_list_matter() {
        let base = hash(&["Person", "City"], &["LIVES_IN"], &["name", "age"]);
        assert_eq!(
            hash(&["City", "Person"], &["LIVES_IN"], &["name", "age"]),
            7_677_116_486_812_619_922
        );
        assert_ne!(
            base,
            hash(&["Person", "City"], &["LIVES_IN"], &["age", "name"])
        );
        // The same name in another list is another schema.
        assert_ne!(hash(&["X"], &[], &[]), hash(&[], &["X"], &[]));
        assert_ne!(hash(&["X"], &[], &[]), hash(&[], &[], &["X"]));
        // The length prefix keeps names from running together.
        assert_ne!(hash(&["ab", "c"], &[], &[]), hash(&["a", "bc"], &[], &[]));
    }

    #[test]
    fn push_matches_of() {
        let mut chain = KeyChain::default();
        chain.push("name");
        chain.push("age");
        assert_eq!(chain, KeyChain::of(["name", "age"]));
    }
}
