//! Batch-mode load CSV operator — streams rows from a CSV file or URL.
//!
//! Implements Cypher `LOAD CSV FROM ...`. For each active row in each input
//! batch, resolves the file path and delimiter, opens a CSV reader, and
//! expands each CSV record into output rows. Output rows are accumulated
//! into batches of up to `BATCH_SIZE`.
//!
//! ```text
//!  Input row ──► eval file path + delimiter
//!                      │
//!             ┌────────▼────────┐
//!             │ file:// path    │ ──► local filesystem (sandboxed to import folder)
//!             │ https:// URL    │ ──► HTTP GET
//!             └────────┬────────┘
//!                      │
//!             ┌────────▼────────┐
//!             │ WITH HEADERS:   │ ──► Map {col_name: value, ...}
//!             │ WITHOUT HEADERS:│ ──► List [field1, field2, ...]
//!             └────────┬────────┘
//!                      │
//!             output rows (one per CSV record)
//! ```

use std::cell::RefCell;
use std::net::{Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    ordermap::OrderMap,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, RowIter};

/// True if `v4` falls into a non-public IPv4 range that we refuse to
/// fetch CSV from. Shared between the `IpAddr::V4` branch and the
/// IPv4-mapped-IPv6 case so an attacker can't bypass the check by
/// publishing a `::ffff:10.0.0.1`-style record.
const fn ipv4_is_forbidden(v4: Ipv4Addr) -> bool {
    let octets = v4.octets();
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_multicast()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_documentation()
        // 100.64.0.0/10 (CGN), 198.18.0.0/15 (benchmark),
        // 192.0.0.0/24 (IETF), 240.0.0.0/4 (reserved)
        || (octets[0] == 100 && (octets[1] & 0xc0) == 0x40)
        || (octets[0] == 198 && (octets[1] & 0xfe) == 0x12)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || octets[0] >= 240
}

/// Validate a remote URL for `LOAD CSV FROM https://...` and return the
/// list of pre-validated `SocketAddr` to connect to.
///
/// Rejects URLs whose DNS resolution includes any non-global address
/// (loopback, link-local, unique-local, multicast, unspecified, or private
/// IPv4 space). This blocks the classic SSRF vectors (cloud metadata,
/// intranet services, localhost scans) without requiring an allow-list.
///
/// The returned `SocketAddr`s are passed to the HTTP client via a custom
/// resolver so the connection is pinned to the exact addresses we
/// validated — closing the TOCTOU window where DNS could otherwise be
/// flipped to a private IP between the validation here and ureq's own
/// resolution.
///
/// Only the `https://` scheme is accepted — callers must have already
/// filtered by prefix.
fn validate_remote_url(url: &str) -> Result<Vec<SocketAddr>, String> {
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| String::from("Only https:// URLs are allowed for LOAD CSV"))?;
    // Strip any path/query/fragment to isolate authority.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() {
        return Err(String::from("URL is missing a host"));
    }
    // Strip userinfo, if any.
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    // Support IPv6 literals in brackets.
    let (host, port) = if let Some(rest) = host_port.strip_prefix('[') {
        let end = rest
            .find(']')
            .ok_or_else(|| String::from("Malformed IPv6 literal in URL"))?;
        let host = &rest[..end];
        let port = rest[end + 1..].strip_prefix(':').unwrap_or("443");
        (host, port)
    } else {
        let mut parts = host_port.rsplitn(2, ':');
        let maybe_port_or_host = parts.next().unwrap_or("");
        parts.next().map_or((maybe_port_or_host, "443"), |host| {
            (host, maybe_port_or_host)
        })
    };
    if host.is_empty() {
        return Err(String::from("URL is missing a host"));
    }
    let port: u16 = port
        .parse()
        .map_err(|_| format!("Invalid port in URL: {port}"))?;

    let addrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("DNS resolution failed for '{host}': {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!("DNS resolution returned no addresses for '{host}'"));
    }
    for addr in &addrs {
        let ip = addr.ip();
        let forbidden = match ip {
            std::net::IpAddr::V4(v4) => ipv4_is_forbidden(v4),
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_multicast()
                    || v6.is_unspecified()
                    // Unique-local fc00::/7 and link-local fe80::/10
                    || (v6.segments()[0] & 0xfe00) == 0xfc00
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
                    // IPv4-mapped: re-use the same predicate as the V4 branch
                    || matches!(v6.to_ipv4_mapped(), Some(m) if ipv4_is_forbidden(m))
            }
        };
        if forbidden {
            return Err(format!(
                "LOAD CSV refused: host '{host}' resolves to a non-public address ({ip})"
            ));
        }
    }
    Ok(addrs)
}

/// Reader that returns `InvalidData` if more than `MAX_CSV_BYTES` are read,
/// instead of silently truncating like `Read::take` does.
struct EnforcingReader<R: std::io::Read> {
    inner: std::io::Take<R>,
    limit: u64,
}

impl<R: std::io::Read> EnforcingReader<R> {
    fn new(
        inner: R,
        limit: u64,
    ) -> Self {
        Self {
            // Read up to limit+1 so we can detect overflow without truncating
            // the legitimate payload.
            inner: std::io::Read::take(inner, limit.saturating_add(1)),
            limit,
        }
    }
}

impl<R: std::io::Read> std::io::Read for EnforcingReader<R> {
    fn read(
        &mut self,
        buf: &mut [u8],
    ) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        // `Take::limit()` returns the number of bytes still allowed to be
        // read; if it's 0 *and* we just read something, the source had at
        // least limit+1 bytes available — i.e. it would have been truncated.
        if n > 0 && self.inner.limit() == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("CSV payload exceeds the {} byte limit", self.limit),
            ));
        }
        Ok(n)
    }
}

/// Resolver used for one LOAD CSV fetch. It returns the pre-validated
/// `SocketAddr`s from `validate_remote_url`, ignoring the URI ureq passes
/// in (this agent is built for exactly one request).
#[derive(Debug)]
struct PinnedResolver {
    addrs: Vec<SocketAddr>,
}

impl ureq::unversioned::resolver::Resolver for PinnedResolver {
    fn resolve(
        &self,
        _uri: &ureq::http::Uri,
        _config: &ureq::config::Config,
        _timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<ureq::unversioned::resolver::ResolvedSocketAddrs, ureq::Error> {
        // `ResolvedSocketAddrs` is a fixed-capacity ArrayVec; its const cap
        // is `MAX_ADDRS = 16`. `validate_remote_url` already runs on a
        // `Vec<SocketAddr>` of arbitrary length, so cap to that limit here
        // to avoid the panic-on-overflow `push`.
        const MAX_ADDRS: usize = 16;
        let mut out = self.empty();
        for addr in self.addrs.iter().take(MAX_ADDRS) {
            out.push(*addr);
        }
        if out.is_empty() {
            Err(ureq::Error::HostNotFound)
        } else {
            Ok(out)
        }
    }
}

/// Build the shared base config (timeouts) once.
fn http_config() -> &'static ureq::config::Config {
    static CFG: OnceLock<ureq::config::Config> = OnceLock::new();
    CFG.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_body(Some(Duration::from_mins(1)))
            .build()
    })
}

/// Owning, lazy CSV record iterator: yields one `Value` per record (a `Map`
/// with headers, a `List` without), parsing from the underlying source on
/// demand — the file's records are never materialized as a whole; the emitter
/// pulls them one at a time while packing.
///
/// A read error mid-stream is parked in the shared `error` slot (the op checks
/// it after every emit and aborts the query) and the iterator fuses.
struct CsvRecordIter {
    records: csv::StringRecordsIntoIter<Box<dyn std::io::Read>>,
    /// `Some(columns)` for `WITH HEADERS` (record → `Value::Map`), `None` for
    /// headerless (record → `Value::List`). Deduplicated once at open time —
    /// unique keys in first-occurrence order, each paired with the column
    /// index its value is read from (the last duplicate occurrence — the
    /// "last wins" replace semantics of building the map per record) — so
    /// every record builds its map with `OrderMap::from_unique_keys`, no
    /// per-record dedup hashing.
    headers: Option<Vec<(Arc<String>, usize)>>,
    /// Shared error slot, also checked on entry so an already-failed load stops
    /// yielding immediately.
    error: Rc<RefCell<Option<String>>>,
}

impl Iterator for CsvRecordIter {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        if self.error.borrow().is_some() {
            return None;
        }
        match self.records.next()? {
            Ok(record) => Some(self.headers.as_ref().map_or_else(
                || {
                    Value::List(Arc::new(
                        record
                            .iter()
                            .map(|field| {
                                if field.is_empty() {
                                    Value::Null
                                } else {
                                    Value::String(Arc::new(String::from(field)))
                                }
                            })
                            .collect(),
                    ))
                },
                |columns| {
                    // Keys are unique by construction (deduplicated once at
                    // open time), so skip per-record dedup hashing. The
                    // reader is non-flexible: any record whose width differs
                    // from the header row errors before reaching here, so
                    // every column index is in bounds and every field is
                    // covered by the plan.
                    Value::Map(Arc::new(OrderMap::from_unique_keys(
                        columns.iter().filter_map(|(name, i)| {
                            record.get(*i).filter(|f| !f.is_empty()).map(|field| {
                                (
                                    Arc::clone(name),
                                    Value::String(Arc::new(String::from(field))),
                                )
                            })
                        }),
                    )))
                },
            )),
            Err(e) => {
                *self.error.borrow_mut() = Some(format!("Failed to read CSV record: {e}"));
                None
            }
        }
    }
}

pub struct LoadCsvOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and performs the shared
    /// pack-and-gather emit, binding each CSV record (a `Value::Map` with
    /// headers, or a `Value::List` without) to `var`. The emitter resumes a
    /// partially-drained batch across `next()` calls, so a CSV that yields more
    /// than `BATCH_SIZE` records never drops sibling input rows.
    pub(crate) emitter: BatchedResultEmitter<'a, Value>,
    file_path: &'a QueryExpr<Variable>,
    headers: &'a bool,
    delimiter: &'a QueryExpr<Variable>,
    /// Shared mid-stream error slot. Records stream lazily through the emitter,
    /// so a CSV read failure can surface while draining (after the row's
    /// iterator was built); the failing iterator parks it here and `next()`
    /// aborts the query on the emit that hit it.
    error: Rc<RefCell<Option<String>>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> LoadCsvOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        file_path: &'a QueryExpr<Variable>,
        headers: &'a bool,
        delimiter: &'a QueryExpr<Variable>,
        var: &'a Variable,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            emitter: BatchedResultEmitter::with_binding(var.id),
            file_path,
            headers,
            delimiter,
            error: Rc::new(RefCell::new(None)),
            idx,
        }
    }

    /// Open the CSV source and return a lazy record iterator. The remote path
    /// downloads the (size-capped) payload eagerly — keeping network I/O inside
    /// the HTTP timeouts — and parses lazily from the buffered bytes; the local
    /// path also reads the file lazily. Either way, records stream one at a
    /// time into the emitter and the parsed record set is never materialized.
    fn open_csv_records(
        headers: bool,
        path: &str,
        delimiter: &Arc<String>,
        error: Rc<RefCell<Option<String>>>,
    ) -> Result<CsvRecordIter, String> {
        // Configurable upper bound for network- and file-sourced CSVs.
        // Kept in sync with prior hardcoded 100 MiB for backward compat.
        const MAX_CSV_BYTES: u64 = 100 * 1024 * 1024;

        let source: Box<dyn std::io::Read> = if path.starts_with("https://") {
            // SEC-1: block SSRF to private / loopback / link-local / multicast
            // hosts by resolving the hostname and inspecting each candidate
            // IP. Only public addresses are permitted for LOAD CSV. The
            // resolved + validated SocketAddrs are pinned into the agent's
            // resolver so ureq cannot independently re-resolve to a
            // different (private) IP between the check and the connect.
            let addrs = validate_remote_url(path)?;

            let agent = ureq::Agent::with_parts(
                http_config().clone(),
                ureq::unversioned::transport::DefaultConnector::new(),
                PinnedResolver { addrs },
            );

            let body = agent
                .get(path)
                .call()
                .map_err(|e| format!("Failed to fetch CSV file: {e}"))?
                .into_body();
            // Enforce the content-length cap to prevent memory-exhaustion DoS.
            // EnforcingReader returns an explicit error rather than silently
            // truncating, so a payload longer than the limit fails the query.
            // The body is downloaded eagerly (lazy network reads would count
            // query processing time against the HTTP body timeout); only the
            // CSV *parsing* streams, from the buffered bytes.
            let mut limited = EnforcingReader::new(body.into_reader(), MAX_CSV_BYTES);
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut limited, &mut bytes)
                .map_err(|e| format!("Failed to fetch CSV file: {e}"))?;
            Box::new(std::io::Cursor::new(bytes))
        } else {
            // SEC-4: cap local file reads at the same bound. The path has
            // already been canonicalised and prefix-checked against the
            // import folder upstream.
            let file =
                std::fs::File::open(path).map_err(|e| format!("Failed to read CSV file: {e}"))?;
            Box::new(EnforcingReader::new(file, MAX_CSV_BYTES))
        };

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(headers)
            .delimiter(delimiter.as_bytes()[0])
            .from_reader(source);

        // The header row is consumed up front (it configures every record's
        // map keys); errors here surface immediately, before any row is packed.
        // Duplicate header names are deduplicated here, once per file: the key
        // keeps its first-occurrence position and reads the last duplicate
        // column's value (the "last wins" semantics of per-record map inserts).
        let header_plan = if headers {
            let header_row = reader
                .headers()
                .map_err(|e| format!("Failed to read CSV headers: {e}"))?;
            let mut columns: Vec<(Arc<String>, usize)> = Vec::with_capacity(header_row.len());
            for (i, name) in header_row.iter().enumerate() {
                if let Some(slot) = columns.iter_mut().find(|(n, _)| n.as_str() == name) {
                    slot.1 = i;
                } else {
                    columns.push((Arc::new(String::from(name)), i));
                }
            }
            Some(columns)
        } else {
            None
        };

        Ok(CsvRecordIter {
            records: reader.into_records(),
            headers: header_plan,
            error,
        })
    }
}

impl<'a> Iterator for LoadCsvOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let runtime = self.runtime;
        let file_path = self.file_path;
        let delimiter_expr = self.delimiter;
        let headers = *self.headers;
        let error = Rc::clone(&self.error);
        loop {
            // For each active input row, resolve the path + delimiter and open a
            // lazy record iterator over the CSV; the emitter pulls records one
            // at a time while packing across rows into gathered batches, so the
            // file's records are never materialized as a whole. The emitter
            // resumes a partially-drained iterator across `next()` calls, so a
            // CSV with more than `BATCH_SIZE` records never drops sibling input
            // rows. When exhausted (`Ok(None)`), pull the next child batch.
            match self.emitter.emit_lazy(|batch, row| {
                // A previous record iterator failed mid-stream: abort instead of
                // opening further files.
                if let Some(e) = error.borrow().as_ref() {
                    return Err(e.clone());
                }
                let view = BatchRow::new(batch, row);
                let path = ExprEval::from_runtime(runtime).eval(
                    file_path,
                    file_path.root().idx(),
                    Some(&view),
                    None,
                )?;
                let Value::String(delimiter) = ExprEval::from_runtime(runtime).eval(
                    delimiter_expr,
                    delimiter_expr.root().idx(),
                    Some(&view),
                    None,
                )?
                else {
                    return Err(String::from("Delimiter must be a string"));
                };
                if delimiter.len() != 1 {
                    return Err(String::from(
                        "CSV field terminator can only be one character wide",
                    ));
                }
                let Value::String(path) = path else {
                    return Err(String::from("File path must be a string"));
                };
                let path = if let Some(path) = path.strip_prefix("file://") {
                    // Strip a leading '/' so an absolute path inside the URL
                    // does not cause `Path::join` to discard the import
                    // folder and escape the sandbox.
                    let rel_path = path.trim_start_matches('/');
                    let joined_path = Path::new(&runtime.import_folder).join(rel_path);
                    let joined = joined_path.to_string_lossy().into_owned();
                    let import_folder =
                        Path::new(&runtime.import_folder)
                            .canonicalize()
                            .map_err(|e| {
                                format!(
                                    "Failed to canonicalize import folder path '{}': {e}",
                                    runtime.import_folder
                                )
                            })?;
                    let cpath = joined_path
                        .canonicalize()
                        .map_err(|e| format!("Failed to canonicalize file path '{joined}': {e}"))?;
                    if !cpath.starts_with(&import_folder) {
                        return Err(format!(
                            "File path '{joined}' is not within the import folder '{}'",
                            runtime.import_folder
                        ));
                    }
                    // Use the canonicalized path for actual I/O so a symlink
                    // race cannot cause us to read a file outside the import
                    // folder between the check and the open.
                    cpath.to_string_lossy().into_owned()
                } else if path.starts_with("https://") {
                    String::from(path.as_str())
                } else {
                    return Err(String::from("File path must start with 'file://' prefix"));
                };

                // Open the CSV and stream its records lazily for this input row.
                let iter = Self::open_csv_records(headers, &path, &delimiter, Rc::clone(&error))?;
                Ok(Some(RowIter::many(Box::new(iter))))
            }) {
                Ok(Some(out)) => {
                    // A record iterator may have failed mid-drain; surface the
                    // error instead of the partially-packed batch.
                    if let Some(e) = self.error.borrow_mut().take() {
                        return Some(Err(e));
                    }
                    return Some(Ok(out));
                }
                Ok(None) => {
                    if let Some(e) = self.error.borrow_mut().take() {
                        return Some(Err(e));
                    }
                    match self.child.next() {
                        Some(Ok(batch)) => self.emitter.seed(batch),
                        Some(Err(e)) => return Some(Err(e)),
                        None => return None,
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
