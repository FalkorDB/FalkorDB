//! v3 as a [`EffectsFormat`]: what a whole payload is, sent and applied.
//!
//! The version-specific half of `effects::EffectsPayload`. It lives here rather
//! than beside the trait so that adding v4 means adding a directory and one arm
//! to the dispatch, and touching nothing else — the trait's module stays free
//! of any single version's rules.

use atomic_refcell::AtomicRefCell;

use super::emit::{build_constraint_buffer, build_index_buffer, for_each_record};
use super::{
    EFFECTS_VERSION, FLAG_COMPRESSED, apply::ApplyError, apply::apply_effects, open_payload, seal,
};
use crate::effects::announce::{AnnouncedConstraint, AnnouncedIndex, SchemaBaseline};
use crate::effects::v3;
use crate::effects::{EffectEncode, EffectWrite};
use crate::effects::{EffectsFormat, EffectsPayload, ReplicationSink};
use crate::graph::graph::Graph;
use crate::runtime::pending::Pending;

impl EffectsFormat<EFFECTS_VERSION> for EffectsPayload {
    fn new_buffer() -> Vec<u8> {
        // Through `v3::new_buffer`, so the flags byte cannot be forgotten.
        v3::new_buffer()
    }

    fn is_empty(buf: &[u8]) -> bool {
        // `u8 version` + `u8 flags`, and nothing after them.
        buf.len() <= HEADER_LEN
    }

    fn replicate(
        sink: &dyn ReplicationSink,
        key: &[u8],
        mut buf: Vec<u8>,
    ) {
        // Compression, and the last thing that touches the bytes. It rewrites
        // everything after the header, so it can only run once and only here —
        // which is why it is not reachable on its own. What "worth compressing"
        // means is `seal`'s: it reads the threshold itself, so the
        // configuration stays behind this boundary too.
        seal(&mut buf);
        sink.replicate(key, &buf);
    }

    fn describe(
        buf: &[u8],
        out: &mut Vec<String>,
    ) {
        // Read straight out of the buffer rather than from `open_payload`,
        // which returns nothing on the failure this is most often called for:
        // a flags byte with a bit v3 does not define is exactly the case where
        // knowing what the byte was matters.
        let flags = buf.get(1).copied().unwrap_or_default();
        let payload = match open_payload(buf) {
            Ok(payload) => payload,
            Err(e) => {
                out.push(format!("v3, flags {flags:#04x}: header unreadable: {e}"));
                return;
            }
        };

        let mut lines = Vec::new();
        let mut decoded = 0usize;
        let mut stopped = false;
        for record in payload.records() {
            match record {
                Ok(record) => {
                    if decoded < DESCRIBE_RECORD_LIMIT {
                        lines.push(format!("record[{decoded}] {record:?}"));
                    }
                    decoded += 1;
                }
                // The iterator fuses here, so this is the last item and
                // `decoded` is the index it failed at.
                Err(e) => {
                    lines.push(format!("record[{decoded}] does not decode: {e}"));
                    stopped = true;
                }
            }
        }
        if decoded > DESCRIBE_RECORD_LIMIT {
            lines.push(format!(
                "… {} more records not shown",
                decoded - DESCRIBE_RECORD_LIMIT
            ));
        }

        let compressed = if flags & FLAG_COMPRESSED == 0 {
            String::new()
        } else {
            format!(", compressed, {} bytes of records", payload.len())
        };
        // Whether the records ran out cleanly is the first thing to know: it
        // separates a payload this build could not *read* from one it read and
        // then refused to apply, and those have different causes.
        let ending = if stopped {
            ", then one that does not"
        } else {
            ", all of which decode"
        };
        out.push(format!(
            "v3, flags {flags:#04x}{compressed}, {decoded} records{ending}"
        ));
        out.extend(lines);
    }

    fn apply(
        graph: &mut Graph,
        buf: &[u8],
    ) -> Result<(), ApplyError> {
        apply_effects(graph, buf)
    }

    fn build<W: EffectWrite + ?Sized>(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        buf: &mut W,
    ) -> Result<u64, String> {
        // The loop itself, not a call to a function that is only this loop.
        // `emit` decides *which* records a commit implies; turning one into
        // bytes is this module's job, and there was nothing in between.
        //
        // `encode` is fallible, and `for_each_record` hands records to a
        // closure that cannot return one, so the first refusal is carried out
        // here. Stopping at the first is deliberate: an `EncodeError` means the
        // emitter built a record wrong, and every later record is written
        // against a buffer that is already not what the replica will be told it
        // is. The write fails rather than shipping a payload with a hole in it.
        let mut n = 0;
        let mut failed = None;
        for_each_record(pending, graph, |record| {
            if failed.is_some() {
                return;
            }
            match record.encode(buf) {
                Ok(()) => n += 1,
                Err(e) => failed = Some(e),
            }
        });
        failed.map_or(Ok(n), |e| Err(e.to_string()))
    }

    fn build_index<W: EffectWrite + ?Sized>(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
        buf: &mut W,
    ) -> Result<(), String> {
        build_index_buffer(pending, graph, create, index, buf)
    }

    fn build_constraint<W: EffectWrite + ?Sized>(
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
        buf: &mut W,
    ) -> Result<(), String> {
        build_constraint_buffer(graph, create, constraint, baseline, buf)
    }
}

/// `u8 version` + `u8 flags`.
const HEADER_LEN: usize = 2;

/// How many records a diverged payload's description names.
///
/// A batched payload can hold thousands, and the ones that matter are at the
/// front and at the break: a record that did not decode is always shown,
/// because the iterator stops there. The count is always reported in full.
const DESCRIBE_RECORD_LIMIT: usize = 16;

#[cfg(test)]
mod tests {
    use crate::effects::v3::{EFFECTS_VERSION, IdList, Record, new_buffer};
    use crate::effects::{EffectEncode, EffectsPayload};

    /// Two records, so a description has something to be truncated from and
    /// the indices mean something.
    fn payload() -> Vec<u8> {
        let mut buf = new_buffer();
        <Record as EffectEncode<EFFECTS_VERSION>>::encode(
            &Record::AddAttribute {
                id: 7,
                name: "name".to_string(),
            },
            &mut buf,
        )
        .unwrap();
        <Record as EffectEncode<EFFECTS_VERSION>>::encode(
            &Record::DeleteNode {
                ids: {
                    let mut ids = IdList::new();
                    ids.push(4);
                    ids.push(5);
                    ids
                },
                labels: vec![1],
            },
            &mut buf,
        )
        .unwrap();
        buf
    }

    #[test]
    fn describe_names_every_record_and_the_bytes() {
        let lines = EffectsPayload::describe(&payload());
        assert_eq!(lines[0], "42 bytes on the wire");
        assert_eq!(lines[1], "v3, flags 0x00, 2 records, all of which decode");
        assert!(lines[2].starts_with("record[0] AddAttribute"), "{lines:?}");
        assert!(lines[3].starts_with("record[1] DeleteNode"), "{lines:?}");
        // The name is in there in the clear, which is half the point of the
        // record rendering.
        assert!(lines[2].contains("\"name\""), "{lines:?}");
        assert_eq!(
            lines[4],
            "bytes[0x0000] 03000a000000070005000000000000006e616d65000500000002000000010001"
        );
        assert_eq!(lines[5], "bytes[0x0020] 00000001000000000402");
    }

    /// The case the record rendering cannot cover on its own, and the reason
    /// the bytes are logged unconditionally.
    #[test]
    fn describe_says_where_decoding_stopped() {
        let mut buf = payload();
        buf.truncate(buf.len() - 3);
        let lines = EffectsPayload::describe(&buf);
        assert_eq!(
            lines[1],
            "v3, flags 0x00, 1 records, then one that does not"
        );
        assert!(lines[2].starts_with("record[0] AddAttribute"), "{lines:?}");
        assert!(
            lines[3].starts_with("record[1] does not decode:"),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("bytes[0x0000] ")),
            "{lines:?}"
        );
    }

    /// A version this build cannot read is exactly when the hex is all there
    /// is, so it still has to be there.
    #[test]
    fn describe_falls_back_to_bytes_for_an_unknown_version() {
        let lines = EffectsPayload::describe(&[99, 0, 0xde, 0xad]);
        assert_eq!(lines[0], "4 bytes on the wire");
        assert_eq!(
            lines[1],
            "version 99: no format in this build reads it, so nothing below is decoded"
        );
        assert_eq!(lines[2], "bytes[0x0000] 6300dead");
    }

    /// A flag bit v3 does not define stops `open_payload` before any record,
    /// and the byte that did it is the thing to report.
    #[test]
    fn describe_reports_an_unreadable_header() {
        let mut buf = payload();
        buf[1] = 0x80;
        let lines = EffectsPayload::describe(&buf);
        assert!(
            lines[1].starts_with("v3, flags 0x80: header unreadable:"),
            "{lines:?}"
        );
    }

    /// Compressed payloads describe their *contents*, not their frame — the
    /// records are what diverged, and the hex still shows what arrived.
    #[test]
    fn describe_decompresses_first() {
        let mut buf = new_buffer();
        for _ in 0..40 {
            <Record as EffectEncode<EFFECTS_VERSION>>::encode(
                &Record::AddAttribute {
                    id: 7,
                    name: "name".to_string(),
                },
                &mut buf,
            )
            .unwrap();
        }
        assert!(crate::effects::v3::maybe_compress(&mut buf, 1));
        let lines = EffectsPayload::describe(&buf);
        assert_eq!(
            lines[1],
            "v3, flags 0x01, compressed, 760 bytes of records, 40 records, all of which decode"
        );
        assert!(lines[2].starts_with("record[0] AddAttribute"), "{lines:?}");
        // And the record cap, which this payload is the only one long enough
        // to reach.
        assert!(lines[17].starts_with("record[15] "), "{lines:?}");
        assert_eq!(lines[18], "… 24 more records not shown");
    }

    #[test]
    fn describe_caps_the_bytes_but_reports_the_length() {
        let mut buf = new_buffer();
        buf.resize(2148, 0);
        let lines = EffectsPayload::describe(&buf);
        assert_eq!(lines[0], "2148 bytes on the wire");
        assert_eq!(lines.last().unwrap(), "… 100 more bytes not shown");
    }
}
