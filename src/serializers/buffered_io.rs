//! Buffered IO layer for RDB serialization (v19 format).
//!
//! Wraps `*mut RedisModuleIO` with a 256KB buffer and prefixes every
//! value with a 1-byte type tag, matching the C FalkorDB `SerializerIOv2`.
//!
//! Type tags:
//! - 0 (BYTES):   `[tag:u8][len:u64][data:len bytes]`
//! - 1 (FLOAT):   `[tag:u8][value:4 bytes]`
//! - 2 (DOUBLE):  `[tag:u8][value:8 bytes]`
//! - 3 (SIGNED):  `[tag:u8][value:8 bytes]`
//! - 4 (UNSIGNED):`[tag:u8][value:8 bytes]`
//! - 5 (LONG_DOUBLE): not used in Rust
//! - 6 (BLOB):    sentinel, next Redis chunk is standalone blob data

use graph::graph::graphblas::serialization::Reader;
use graph::graph::graphblas::serialization::Writer;
use redis_module::RedisModuleIO;
use redis_module::raw;

const BUFFER_SIZE: usize = 256_000;

const TYPE_BYTES: u8 = 0;
const TYPE_FLOAT: u8 = 1;
const TYPE_DOUBLE: u8 = 2;
const TYPE_SIGNED: u8 = 3;
const TYPE_UNSIGNED: u8 = 4;
#[allow(dead_code)]
const TYPE_LONG_DOUBLE: u8 = 5;
const TYPE_BLOB: u8 = 6;

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Buffered writer that accumulates type-tagged values and flushes
/// as 256KB chunks to Redis via `RedisModule_SaveStringBuffer`.
pub struct BufferedWriter {
    rdb: *mut RedisModuleIO,
    buf: Vec<u8>,
}

impl BufferedWriter {
    pub fn new(rdb: *mut RedisModuleIO) -> Self {
        Self {
            rdb,
            buf: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    /// Flush the current buffer to Redis and reset.
    fn flush(&mut self) {
        if !self.buf.is_empty() {
            raw::save_slice(self.rdb, &self.buf);
            self.buf.clear();
        }
    }

    /// Ensure there is room for `needed` bytes, flushing if necessary.
    fn accommodate(
        &mut self,
        needed: usize,
    ) {
        if self.buf.len() + needed > BUFFER_SIZE {
            self.flush();
        }
    }

    pub fn write_unsigned(
        &mut self,
        val: u64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_UNSIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    pub fn write_signed(
        &mut self,
        val: i64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_SIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    pub fn write_double(
        &mut self,
        val: f64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_DOUBLE);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    #[allow(dead_code)]
    pub fn write_float(
        &mut self,
        val: f32,
    ) {
        self.accommodate(1 + 4);
        self.buf.push(TYPE_FLOAT);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    /// Write a byte buffer. Small buffers are inlined; large ones use
    /// the blob sentinel and are written as standalone Redis chunks.
    pub fn write_buffer(
        &mut self,
        data: &[u8],
    ) {
        let inline_size = 1 + 8 + data.len(); // tag + u64 len + data
        if inline_size <= BUFFER_SIZE {
            // Inline: fits in a single buffer
            self.accommodate(inline_size);
            self.buf.push(TYPE_BYTES);
            self.buf
                .extend_from_slice(&(data.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(data);
        } else {
            // Blob: write sentinel, flush, then write standalone
            self.accommodate(1);
            self.buf.push(TYPE_BLOB);
            self.flush();
            raw::save_slice(self.rdb, data);
        }
    }

    /// Flush any remaining data. Must be called when encoding is complete.
    pub fn finish(mut self) {
        self.flush();
    }
}

impl Writer for BufferedWriter {
    fn write_unsigned(
        &mut self,
        val: u64,
    ) {
        self.write_unsigned(val);
    }

    fn write_signed(
        &mut self,
        val: i64,
    ) {
        self.write_signed(val);
    }

    fn write_double(
        &mut self,
        val: f64,
    ) {
        self.write_double(val);
    }

    fn write_buffer(
        &mut self,
        data: &[u8],
    ) {
        self.write_buffer(data);
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Buffered reader that loads 256KB chunks from Redis and consumes
/// type-tagged values from them.
pub struct BufferedReader {
    rdb: *mut RedisModuleIO,
    buf: Vec<u8>,
    pos: usize,
}

impl Reader for BufferedReader {
    fn read_unsigned(&mut self) -> Result<u64, String> {
        self.read_unsigned()
    }

    fn read_signed(&mut self) -> Result<i64, String> {
        self.read_signed()
    }

    fn read_double(&mut self) -> Result<f64, String> {
        self.read_double()
    }

    fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
        self.read_buffer()
    }
}

impl BufferedReader {
    pub const fn new(rdb: *mut RedisModuleIO) -> Self {
        Self {
            rdb,
            buf: Vec::new(),
            pos: 0,
        }
    }

    pub const fn from_vec(data: Vec<u8>) -> Self {
        Self {
            rdb: std::ptr::null_mut(),
            buf: data,
            pos: 0,
        }
    }

    /// Load the next chunk from Redis.
    fn load_chunk(&mut self) -> Result<(), String> {
        let chunk = raw::load_string_buffer(self.rdb)
            .map_err(|e| format!("BufferedReader: load chunk: {e}"))?;
        self.buf = chunk.as_ref().to_vec();
        self.pos = 0;
        Ok(())
    }

    /// Ensure at least 1 byte is available, loading a new chunk if needed.
    fn ensure_available(&mut self) -> Result<(), String> {
        if self.pos >= self.buf.len() {
            self.load_chunk()?;
        }
        Ok(())
    }

    /// Read and validate a type tag byte.
    fn read_tag(
        &mut self,
        expected: u8,
    ) -> Result<(), String> {
        self.ensure_available()?;
        let tag = self.buf[self.pos];
        self.pos += 1;
        if tag != expected {
            return Err(format!(
                "BufferedReader: expected type tag {expected}, got {tag} at pos {}",
                self.pos - 1
            ));
        }
        Ok(())
    }

    /// Read N bytes from the buffer.
    fn read_bytes(
        &mut self,
        n: usize,
    ) -> Result<&[u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!(
                "BufferedReader: need {n} bytes at pos {}, but buffer len is {}",
                self.pos,
                self.buf.len()
            ));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn read_unsigned(&mut self) -> Result<u64, String> {
        self.read_tag(TYPE_UNSIGNED)?;
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_signed(&mut self) -> Result<i64, String> {
        self.read_tag(TYPE_SIGNED)?;
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_double(&mut self) -> Result<f64, String> {
        self.read_tag(TYPE_DOUBLE)?;
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    #[allow(dead_code)]
    pub fn read_float(&mut self) -> Result<f32, String> {
        self.read_tag(TYPE_FLOAT)?;
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Read a byte buffer. Handles both inline (TYPE_BYTES) and blob (TYPE_BLOB).
    pub fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
        self.ensure_available()?;
        let tag = self.buf[self.pos];
        self.pos += 1;

        match tag {
            TYPE_BYTES => {
                // Inline: length then data
                let len_bytes = self.read_bytes(8)?;
                let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
                let data = self.read_bytes(len)?;
                Ok(data.to_vec())
            }
            TYPE_BLOB => {
                // The current buffer should now be fully consumed
                // (the blob sentinel was the last byte before flush).
                // Load the standalone blob chunk.
                let chunk = raw::load_string_buffer(self.rdb)
                    .map_err(|e| format!("BufferedReader: load blob: {e}"))?;
                let data = chunk.as_ref().to_vec();
                // Reset internal state - next read will trigger load_chunk
                self.buf.clear();
                self.pos = 0;
                Ok(data)
            }
            _ => Err(format!(
                "BufferedReader: expected BYTES(0) or BLOB(6) tag, got {tag}"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Pipe-based Writer / Reader for GRAPH.COPY (fork + pipe)
// ---------------------------------------------------------------------------

use std::os::raw::c_int;

unsafe extern "C" {
    fn write(
        fd: c_int,
        buf: *const std::ffi::c_void,
        count: usize,
    ) -> isize;
    fn read(
        fd: c_int,
        buf: *mut std::ffi::c_void,
        count: usize,
    ) -> isize;
    fn close(fd: c_int) -> c_int;
}

/// Buffered writer that sends type-tagged values through a pipe fd.
///
/// Each flush writes a length-prefixed chunk: `[len:u64 LE][data:len bytes]`.
/// A zero-length chunk signals end-of-stream.
pub struct PipeWriter {
    fd: i32,
    buf: Vec<u8>,
}

impl PipeWriter {
    pub fn new(fd: i32) -> Self {
        Self {
            fd,
            buf: Vec::with_capacity(BUFFER_SIZE),
        }
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            let len = self.buf.len() as u64;
            write_all_fd(self.fd, &len.to_le_bytes());
            write_all_fd(self.fd, &self.buf);
            self.buf.clear();
        }
    }

    fn accommodate(
        &mut self,
        needed: usize,
    ) {
        if self.buf.len() + needed > BUFFER_SIZE {
            self.flush();
        }
    }

    /// Flush remaining data, send zero-length terminator, and close the fd.
    pub fn finish(mut self) {
        self.flush();
        // Zero-length chunk signals end-of-stream.
        let zero = 0u64;
        write_all_fd(self.fd, &zero.to_le_bytes());
        unsafe { close(self.fd) };
    }
}

impl Writer for PipeWriter {
    fn write_unsigned(
        &mut self,
        val: u64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_UNSIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_signed(
        &mut self,
        val: i64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_SIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_double(
        &mut self,
        val: f64,
    ) {
        self.accommodate(1 + 8);
        self.buf.push(TYPE_DOUBLE);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_buffer(
        &mut self,
        data: &[u8],
    ) {
        let inline_size = 1 + 8 + data.len();
        if inline_size <= BUFFER_SIZE {
            self.accommodate(inline_size);
            self.buf.push(TYPE_BYTES);
            self.buf
                .extend_from_slice(&(data.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(data);
        } else {
            // Blob: write sentinel, flush, then send standalone chunk.
            self.accommodate(1);
            self.buf.push(TYPE_BLOB);
            self.flush();
            let len = data.len() as u64;
            write_all_fd(self.fd, &len.to_le_bytes());
            write_all_fd(self.fd, data);
        }
    }
}

/// Buffered reader that receives type-tagged values from a pipe fd.
pub struct PipeReader {
    fd: i32,
    buf: Vec<u8>,
    pos: usize,
}

impl Reader for PipeReader {
    fn read_unsigned(&mut self) -> Result<u64, String> {
        self.read_tag(TYPE_UNSIGNED)?;
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_signed(&mut self) -> Result<i64, String> {
        self.read_tag(TYPE_SIGNED)?;
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_double(&mut self) -> Result<f64, String> {
        self.read_tag(TYPE_DOUBLE)?;
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
        self.ensure_available()?;
        let tag = self.buf[self.pos];
        self.pos += 1;

        match tag {
            TYPE_BYTES => {
                let len_bytes = self.read_bytes(8)?;
                let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
                let data = self.read_bytes(len)?;
                Ok(data.to_vec())
            }
            TYPE_BLOB => {
                self.load_chunk()?;
                let data = self.buf[self.pos..].to_vec();
                self.pos = self.buf.len();
                Ok(data)
            }
            _ => Err(format!(
                "PipeReader: expected BYTES(0) or BLOB(6) tag, got {tag}"
            )),
        }
    }
}

impl PipeReader {
    pub fn new(fd: i32) -> Self {
        Self {
            fd,
            buf: Vec::new(),
            pos: 0,
        }
    }

    /// Load the next length-prefixed chunk from the pipe.
    /// Returns Err if the stream ended (zero-length chunk).
    fn load_chunk(&mut self) -> Result<(), String> {
        let mut len_buf = [0u8; 8];
        read_all_fd(self.fd, &mut len_buf)?;
        let len = u64::from_le_bytes(len_buf) as usize;
        if len == 0 {
            return Err("PipeReader: end of stream".to_string());
        }
        self.buf.resize(len, 0);
        read_all_fd(self.fd, &mut self.buf)?;
        self.pos = 0;
        Ok(())
    }

    fn ensure_available(&mut self) -> Result<(), String> {
        if self.pos >= self.buf.len() {
            self.load_chunk()?;
        }
        Ok(())
    }

    fn read_tag(
        &mut self,
        expected: u8,
    ) -> Result<(), String> {
        self.ensure_available()?;
        let tag = self.buf[self.pos];
        self.pos += 1;
        if tag != expected {
            return Err(format!(
                "PipeReader: expected type tag {expected}, got {tag} at pos {}",
                self.pos - 1
            ));
        }
        Ok(())
    }

    fn read_bytes(
        &mut self,
        n: usize,
    ) -> Result<&[u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!(
                "PipeReader: need {n} bytes at pos {}, but buffer len is {}",
                self.pos,
                self.buf.len()
            ));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Close the underlying pipe fd.
    pub fn close(self) {
        unsafe { close(self.fd) };
    }
}

// ---------------------------------------------------------------------------
// Vec-based Writer / Reader for GRAPH.RESTORE replication
// ---------------------------------------------------------------------------

/// Writer that appends type-tagged values to a `Vec<u8>`.
///
/// Same tag format as `BufferedWriter` but no chunking and no BLOB sentinel
/// (always inlines buffers as TYPE_BYTES). Readable by `BufferedReader::from_vec()`.
pub struct VecWriter {
    buf: Vec<u8>,
}

impl VecWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consume the writer and return the serialized bytes.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

impl Writer for VecWriter {
    fn write_unsigned(
        &mut self,
        val: u64,
    ) {
        self.buf.push(TYPE_UNSIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_signed(
        &mut self,
        val: i64,
    ) {
        self.buf.push(TYPE_SIGNED);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_double(
        &mut self,
        val: f64,
    ) {
        self.buf.push(TYPE_DOUBLE);
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_buffer(
        &mut self,
        data: &[u8],
    ) {
        self.buf.push(TYPE_BYTES);
        self.buf
            .extend_from_slice(&(data.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(data);
    }
}

/// Write all bytes to a file descriptor, retrying on EINTR.
fn write_all_fd(
    fd: i32,
    mut data: &[u8],
) {
    while !data.is_empty() {
        let n = unsafe { write(fd, data.as_ptr().cast(), data.len()) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            // In fork child, best-effort — just stop.
            return;
        }
        data = &data[n as usize..];
    }
}

/// Read exactly `buf.len()` bytes from a file descriptor, retrying on EINTR.
fn read_all_fd(
    fd: i32,
    buf: &mut [u8],
) -> Result<(), String> {
    let mut offset = 0;
    while offset < buf.len() {
        let n = unsafe { read(fd, buf[offset..].as_mut_ptr().cast(), buf.len() - offset) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(format!("PipeReader: read error: {err}"));
        }
        if n == 0 {
            return Err("PipeReader: unexpected EOF".to_string());
        }
        offset += n as usize;
    }
    Ok(())
}
