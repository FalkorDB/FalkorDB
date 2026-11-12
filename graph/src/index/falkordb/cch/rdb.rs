//! The CCH definitions payload of an RDB (`EncodeState::CchIndices`, encoding version 20).
//!
//! Definition only, as C writes it; the hierarchy is rebuilt on load.

use super::CchIndexDef;
use crate::graph::{
    graph::TypeId,
    graphblas::serialization::{Reader, Writer},
};

/// Writes CCH index definitions as C's `RdbSaveCCH_v20` / `CCHIndex_RdbSave` do: the index
/// count, then per index the relationship-type count, each relationship-type id, and the weight
/// attribute id, all as unsigned values. Definition only; the hierarchy is rebuilt on load.
pub(crate) fn encode_cch_defs(
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
mod tests {
    use super::*;
    use crate::graph::graphblas::serialization::{EncodeState, PayloadEntry};
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

    /// A build without the feature reads past the payload with `skip_cch_defs` instead of
    /// decoding it: both must consume exactly what the encoder wrote and refuse the same
    /// malformed streams, or that build would misread the payloads after it.
    #[test]
    fn the_skip_without_the_feature_reads_what_the_decoder_reads() {
        use crate::graph::graphblas::serialization::skip_cch_defs;
        const NEXT: u64 = 0x00C0_FFEE;
        let defs = [
            CchIndexDef::new(vec![TypeId(1), TypeId(4)], 7),
            CchIndexDef::new(vec![TypeId(2)], 0),
        ];
        let mut s = Stream::default();
        encode_cch_defs(&mut s, &defs);
        s.0.push_back(NEXT);
        let mut skipped = Stream(s.0.clone());
        assert_eq!(decode_cch_defs(&mut s).unwrap(), defs);
        assert_eq!(skip_cch_defs(&mut skipped), Ok(2));
        assert_eq!(s.0, skipped.0, "both stop on the next payload");
        for bad in [vec![1], vec![1, 0, 3], vec![1, 1, 2], vec![1, 1, 2, 70000]] {
            assert!(
                decode_cch_defs(&mut Stream(bad.clone().into())).is_err(),
                "decode {bad:?}"
            );
            assert!(
                skip_cch_defs(&mut Stream(bad.clone().into())).is_err(),
                "skip {bad:?}"
            );
        }
    }

    #[test]
    fn graph_encodes_its_defs() {
        use crate::{
            graph::{graph::Graph, graphblas::test_init},
            index::falkordb::{
                cch::{
                    Cch, CchIndex,
                    testkit::{grid, to_input},
                },
                versioned::{BuildState, IndexEntry},
            },
        };
        use std::sync::Arc;
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_rdb_test");
        let def = CchIndexDef::new(vec![TypeId(3), TypeId(1)], 5);
        g.falkordb_index.cch_indexes.insert(
            def.clone(),
            IndexEntry::new(
                CchIndex::built(
                    def.clone(),
                    Arc::new(Cch::build(&to_input(&grid(2, 2))).expect("build")),
                ),
                BuildState::Ready,
                // Any generation: nothing here compares it.
                1,
            ),
        );
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
