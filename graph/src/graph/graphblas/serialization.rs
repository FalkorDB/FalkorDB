//! Serialization traits and type tags for RDB persistence.
//!
//! Provides `Writer`/`Reader` traits, `Encode`/`Decode` traits, and
//! type-tag modules used by the encoder/decoder in the `serializers`
//! module which handles the actual Redis Module IO.

use roaring::RoaringTreemap;

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
    fn decode_with_count(
        &mut self,
        r: &mut dyn Reader,
        count: u64,
    ) -> Result<(), String> {
        let _ = (r, count);
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
pub mod si_type {
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
    Final = 10,
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
            10 => Some(Self::Final),
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
