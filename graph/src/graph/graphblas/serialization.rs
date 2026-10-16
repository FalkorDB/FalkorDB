//! Serialization traits and type tags for RDB persistence.
//!
//! Provides `Writer`/`Reader` traits, `Encode`/`Decode` traits, and
//! type-tag modules used by the encoder/decoder in the `serializers`
//! module which handles the actual Redis Module IO.

use roaring::RoaringTreemap;

use crate::{graph::graph::TypeId, index::cch::CchIndexDef};

/// Abstraction over a serialization sink.
///
/// The root crate implements this for `BufferedWriter` (v19 buffered IO).
/// The graph crate uses it via `Encode` impls without knowing about Redis.
pub trait Writer {
    fn write_unsigned(
        &mut self,
        val: u64,
    );
    fn write_signed(
        &mut self,
        val: i64,
    );
    fn write_double(
        &mut self,
        val: f64,
    );
    fn write_buffer(
        &mut self,
        data: &[u8],
    );
}

/// Types that can serialize themselves into a [`Writer`].
pub trait Encode<const VERSION: u64> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    );

    /// Encode a range of entities starting at `offset`, encoding `count` items.
    fn encode_with_range(
        &self,
        w: &mut dyn Writer,
        count: u64,
        offset: u64,
    ) {
        let _ = (w, count, offset);
        unimplemented!()
    }
}

/// Abstraction over a deserialization source.
///
/// The root crate implements this for `BufferedReader` (v19 buffered IO).
/// The graph crate uses it via `Decode` impls without knowing about Redis.
pub trait Reader {
    fn read_unsigned(&mut self) -> Result<u64, String>;
    fn read_signed(&mut self) -> Result<i64, String>;
    fn read_double(&mut self) -> Result<f64, String>;
    fn read_buffer(&mut self) -> Result<Vec<u8>, String>;
}

/// Types that can deserialize themselves from a [`Reader`].
pub trait Decode<const VERSION: u64>: Sized {
    fn decode(r: &mut dyn Reader) -> Result<Self, String>;

    /// Decode `count` entities from the reader into `self`.
    ///
    /// `attr_limit` is the length of the graph's attribute dictionary: any
    /// attribute id at or above it names nothing and must be dropped. It rides on
    /// the trait rather than on an inherent method so the bound cannot be skipped
    /// — an earlier revision had `AttributeStore` satisfy this signature by
    /// assuming `usize::MAX`, which silently disabled the check on two of the
    /// three load paths, including the multi-key one (every graph large enough to
    /// be split across virtual keys). Implementors that decode no attributes
    /// ignore it.
    fn decode_with_count(
        &mut self,
        r: &mut dyn Reader,
        count: u64,
        attr_limit: usize,
    ) -> Result<(), String> {
        let _ = (r, count, attr_limit);
        unimplemented!()
    }
}

/// Index field type bitmask matching C FalkorDB index_field.h.
pub mod index_field_type {
    pub const INDEX_FLD_FULLTEXT: u64 = 0x01;
    pub const INDEX_FLD_NUMERIC: u64 = 0x02;
    pub const INDEX_FLD_GEO: u64 = 0x04;
    pub const INDEX_FLD_STR: u64 = 0x08;
    pub const INDEX_FLD_VECTOR: u64 = 0x10;
}

/// SIValue type tags for binary serialization (matching C FalkorDB format).
///
/// C's `SIType` is a **bitmask**, not an ordinal: each type is a distinct bit,
/// which is why these are shifts rather than 0..12. Rust's own v2 effects codec
/// used sequential tags, and they collide with these almost everywhere.
///
/// `u64`, because the RDB writes them through `write_unsigned`, which takes a
/// `u64`. The effects wire carries a **four-byte** tag — the width C declares
/// for `SIType` and what both v2 and v3 put on the wire — so that path narrows
/// once, in `effects::v3::wire_tag`, where a test proves every constant here
/// survives the narrowing.
pub mod si_type {
    pub const T_MAP: u64 = 1 << 0;
    pub const T_ARRAY: u64 = 1 << 3;
    pub const T_DATETIME: u64 = 1 << 5;
    pub const T_DATE: u64 = 1 << 7;
    pub const T_TIME: u64 = 1 << 8;
    pub const T_DURATION: u64 = 1 << 10;
    pub const T_STRING: u64 = 1 << 11;
    pub const T_BOOL: u64 = 1 << 12;
    pub const T_INT64: u64 = 1 << 13;
    pub const T_DOUBLE: u64 = 1 << 14;
    pub const T_NULL: u64 = 1 << 15;
    pub const T_POINT: u64 = 1 << 17;
    pub const T_VECTOR_F32: u64 = 1 << 18;
    pub const T_INTERN: u64 = 1 << 19;
}

/// Identifies which payload section a key entry represents in the RDB format.
///
/// Each virtual key stores a directory of `(EncodeState, count)` pairs describing
/// which payload sections it contains and how many entities per section.
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeState {
    Init = 0,
    Nodes = 1,
    DeletedNodes = 2,
    Edges = 3,
    DeletedEdges = 4,
    GraphSchema = 5,
    LabelsMatrices = 6,
    RelationMatrices = 7,
    AdjMatrix = 8,
    LblsMatrix = 9,
    /// Graph-level CCH path-index definitions (definition only; hierarchies are rebuilt on load).
    CchIndices = 10,
    /// Never written to an RDB.
    Final = 11,
}

impl EncodeState {
    #[must_use]
    pub const fn from_u64(v: u64) -> Option<Self> {
        match v {
            0 => Some(Self::Init),
            1 => Some(Self::Nodes),
            2 => Some(Self::DeletedNodes),
            3 => Some(Self::Edges),
            4 => Some(Self::DeletedEdges),
            5 => Some(Self::GraphSchema),
            6 => Some(Self::LabelsMatrices),
            7 => Some(Self::RelationMatrices),
            8 => Some(Self::AdjMatrix),
            9 => Some(Self::LblsMatrix),
            10 => Some(Self::CchIndices),
            11 => Some(Self::Final),
            _ => None,
        }
    }
}

/// A single payload entry with state, count, and offset into the entity stream.
#[derive(Debug, Clone, Copy)]
pub struct PayloadEntry {
    pub state: EncodeState,
    pub count: u64,
    pub offset: u64,
}

impl Encode<19> for RoaringTreemap {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        self.encode_with_range(w, self.len(), 0);
    }

    fn encode_with_range(
        &self,
        w: &mut dyn Writer,
        count: u64,
        offset: u64,
    ) {
        let mut buf = Vec::with_capacity(count as usize * 8);
        for id in self.iter().skip(offset as usize).take(count as usize) {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        w.write_buffer(&buf);
    }
}

impl Decode<19> for RoaringTreemap {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let bytes = r.read_buffer()?;
        if bytes.len() % 8 != 0 {
            return Err(format!(
                "misaligned deleted entities buffer: {} bytes is not a multiple of 8",
                bytes.len()
            ));
        }
        let count = bytes.len() / 8;
        let mut bitmap = Self::new();
        for i in 0..count {
            let id = u64::from_le_bytes(
                bytes[i * 8..(i + 1) * 8]
                    .try_into()
                    .map_err(|_| "invalid id bytes")?,
            );
            bitmap.insert(id);
        }
        Ok(bitmap)
    }

    fn decode_with_count(
        &mut self,
        r: &mut dyn Reader,
        count: u64,
        _attr_limit: usize,
    ) -> Result<(), String> {
        let bytes = r.read_buffer()?;
        let expected_len = count as usize * 8;
        if bytes.len() != expected_len {
            return Err(format!(
                "deleted entities buffer length mismatch: got {} bytes, expected {} bytes",
                bytes.len(),
                expected_len
            ));
        }
        for i in 0..count as usize {
            let id = u64::from_le_bytes(
                bytes[i * 8..(i + 1) * 8]
                    .try_into()
                    .map_err(|_| "invalid id bytes")?,
            );
            self.insert(id);
        }
        Ok(())
    }
}

/// Writes CCH index definitions as C's `RdbSaveCCH_v20` / `CCHIndex_RdbSave` do: the index
/// count, then per index the relationship-type count, each relationship-type id, and the weight
/// attribute id, all as unsigned values. Definition only; the hierarchy is rebuilt on load.
pub fn encode_cch_defs(
    w: &mut dyn Writer,
    defs: &[CchIndexDef],
) {
    w.write_unsigned(defs.len() as u64);
    for def in defs {
        w.write_unsigned(def.rel_types().len() as u64);
        for t in def.rel_types() {
            w.write_unsigned(t.0 as u64);
        }
        w.write_unsigned(u64::from(def.weight_attr));
    }
}

/// Reads what [`encode_cch_defs`] writes. Rebuilds each definition through
/// `CchIndexDef::new`, so the sorted/deduplicated invariant holds whatever the stream order was.
///
/// # Errors
/// A truncated stream, an index with no relationship types, or a weight attribute id above `u16::MAX`.
pub fn decode_cch_defs(r: &mut dyn Reader) -> Result<Vec<CchIndexDef>, String> {
    let count = r.read_unsigned()?;
    let mut defs = Vec::new();
    for _ in 0..count {
        let n = r.read_unsigned()?;
        if n == 0 {
            return Err("CCH index with no relationship types".to_string());
        }
        let mut types = Vec::new();
        for _ in 0..n {
            types.push(TypeId(r.read_unsigned()? as usize));
        }
        let attr = r.read_unsigned()?;
        let attr = u16::try_from(attr)
            .map_err(|_| format!("CCH weight attribute id {attr} out of range"))?;
        defs.push(CchIndexDef::new(types, attr));
    }
    Ok(defs)
}

#[cfg(test)]
mod cch_def_tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct Stream(VecDeque<u64>);
    impl Writer for Stream {
        fn write_unsigned(
            &mut self,
            val: u64,
        ) {
            self.0.push_back(val);
        }
        fn write_signed(
            &mut self,
            _: i64,
        ) {
            unreachable!()
        }
        fn write_double(
            &mut self,
            _: f64,
        ) {
            unreachable!()
        }
        fn write_buffer(
            &mut self,
            _: &[u8],
        ) {
            unreachable!()
        }
    }
    impl Reader for Stream {
        fn read_unsigned(&mut self) -> Result<u64, String> {
            self.0.pop_front().ok_or_else(|| "truncated".to_string())
        }
        fn read_signed(&mut self) -> Result<i64, String> {
            unreachable!()
        }
        fn read_double(&mut self) -> Result<f64, String> {
            unreachable!()
        }
        fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
            unreachable!()
        }
    }

    #[test]
    fn matches_the_c_stream_layout() {
        // C: count, then per index n, n rel ids, weight attr -- hand-written, not from our writer.
        let defs = [
            CchIndexDef::new(vec![TypeId(1), TypeId(4)], 7),
            CchIndexDef::new(vec![TypeId(2)], 0),
        ];
        let mut s = Stream::default();
        encode_cch_defs(&mut s, &defs);
        assert_eq!(s.0, [2, 2, 1, 4, 7, 1, 2, 0]);
    }

    #[test]
    fn round_trips_and_normalises() {
        let mut s = Stream::default();
        // unsorted, duplicated rel ids as a foreign writer might produce
        s.0.extend([1, 3, 5, 2, 5, 9]);
        let defs = decode_cch_defs(&mut s).unwrap();
        assert_eq!(defs, vec![CchIndexDef::new(vec![TypeId(2), TypeId(5)], 9)]);
        assert!(s.0.is_empty());
        let mut out = Stream::default();
        encode_cch_defs(&mut out, &defs);
        assert_eq!(out.0, [1, 2, 2, 5, 9]);
    }

    #[test]
    fn refuses_malformed_streams() {
        for bad in [vec![1], vec![1, 0, 3], vec![1, 1, 2], vec![1, 1, 2, 70000]] {
            assert!(
                decode_cch_defs(&mut Stream(bad.clone().into())).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn state_numbers_match_c() {
        assert_eq!(EncodeState::CchIndices as u64, 10);
        assert_eq!(EncodeState::Final as u64, 11);
        assert!(matches!(
            EncodeState::from_u64(10),
            Some(EncodeState::CchIndices)
        ));
        assert!(matches!(
            EncodeState::from_u64(9),
            Some(EncodeState::LblsMatrix)
        ));
        assert!(EncodeState::from_u64(12).is_none());
    }

    #[test]
    fn graph_encodes_its_defs_and_a_graph_without_any_has_none() {
        use crate::{
            graph::{graph::Graph, graphblas::test_init},
            index::cch::{
                Cch, CchIndex,
                testkit::{grid, to_input},
            },
        };
        use std::sync::Arc;
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_rdb_test");
        assert!(
            g.cch_indexes().next().is_none(),
            "no index, no CchIndices payload"
        );

        let def = CchIndexDef::new(vec![TypeId(3), TypeId(1)], 5);
        g.cch_indexes.push(CchIndex {
            def: def.clone(),
            cch: Arc::new(Cch::build(&to_input(&grid(2, 2))).expect("build")),
        });
        let mut s = Stream::default();
        let p = PayloadEntry {
            state: EncodeState::CchIndices,
            count: 1,
            offset: 0,
        };
        g.encode_payload(&mut s, &p);
        assert_eq!(s.0, [1, 2, 1, 3, 5]);
        assert_eq!(decode_cch_defs(&mut s).unwrap(), vec![def]);
    }
}
