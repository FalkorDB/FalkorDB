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

use std::collections::VecDeque;
use std::net::ToSocketAddrs;
use std::path::Path;
use std::sync::Arc;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    ordermap::OrderMap,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Validate a remote URL for `LOAD CSV FROM https://...`.
///
/// Rejects URLs whose DNS resolution includes any non-global address
/// (loopback, link-local, unique-local, multicast, unspecified, or private
/// IPv4 space). This blocks the classic SSRF vectors (cloud metadata,
/// intranet services, localhost scans) without requiring an allow-list.
///
/// Only the `https://` scheme is accepted — callers must have already
/// filtered by prefix.
fn validate_remote_url(url: &str) -> Result<(), String> {
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

    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("DNS resolution failed for '{host}': {e}"))?;
    let mut saw_any = false;
    for addr in addrs {
        saw_any = true;
        let ip = addr.ip();
        let forbidden = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_multicast()
                    || v4.is_broadcast()
                    || v4.is_unspecified()
                    || v4.is_documentation()
                    // 100.64.0.0/10 (CGN), 198.18.0.0/15 (benchmark),
                    // 192.0.0.0/24 (IETF), 240.0.0.0/4 (reserved)
                    || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 0x40)
                    || (v4.octets()[0] == 198 && (v4.octets()[1] & 0xfe) == 0x12)
                    || (v4.octets()[0] == 192
                        && v4.octets()[1] == 0
                        && v4.octets()[2] == 0)
                    || v4.octets()[0] >= 240
            }
            std::net::IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_multicast()
                    || v6.is_unspecified()
                    // Unique-local fc00::/7 and link-local fe80::/10
                    || (v6.segments()[0] & 0xfe00) == 0xfc00
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
                    // IPv4-mapped: inspect the mapped v4 using recursion-free inline rules
                    || matches!(v6.to_ipv4_mapped(), Some(m) if m.is_loopback() || m.is_private() || m.is_link_local())
            }
        };
        if forbidden {
            return Err(format!(
                "LOAD CSV refused: host '{host}' resolves to a non-public address ({ip})"
            ));
        }
    }
    if !saw_any {
        return Err(format!("DNS resolution returned no addresses for '{host}'"));
    }
    Ok(())
}

pub struct LoadCsvOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<Env<'a>>,
    file_path: &'a QueryExpr<Variable>,
    headers: &'a bool,
    delimiter: &'a QueryExpr<Variable>,
    var: &'a Variable,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> LoadCsvOp<'a> {
    pub const fn new(
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
            pending: VecDeque::new(),
            file_path,
            headers,
            delimiter,
            var,
            idx,
        }
    }

    fn load_csv_records(
        &self,
        path: &str,
        delimiter: &Arc<String>,
        vars: &Env<'a>,
    ) -> Result<Vec<Env<'a>>, String> {
        let mut results = Vec::new();

        // Configurable upper bound for network- and file-sourced CSVs.
        // Kept in sync with prior hardcoded 100 MiB for backward compat.
        const MAX_CSV_BYTES: u64 = 100 * 1024 * 1024;

        if path.starts_with("https://") {
            // SEC-1: block SSRF to private / loopback / link-local / multicast
            // hosts by resolving the hostname and inspecting each candidate
            // IP. Only public addresses are permitted for LOAD CSV.
            validate_remote_url(path)?;

            let body = ureq::get(path)
                .call()
                .map_err(|e| format!("Failed to fetch CSV file: {e}"))?
                .into_body();
            // Enforce content-length cap to prevent memory-exhaustion DoS.
            let response = std::io::Read::take(body.into_reader(), MAX_CSV_BYTES);
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(*self.headers)
                .delimiter(delimiter.as_bytes()[0])
                .from_reader(response);
            self.collect_records(&mut reader, vars, &mut results)?;
        } else {
            // SEC-4: cap local file reads at the same bound. The path has
            // already been canonicalised and prefix-checked against the
            // import folder upstream.
            let file =
                std::fs::File::open(path).map_err(|e| format!("Failed to read CSV file: {e}"))?;
            let limited = std::io::Read::take(file, MAX_CSV_BYTES);
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(*self.headers)
                .delimiter(delimiter.as_bytes()[0])
                .from_reader(limited);
            self.collect_records(&mut reader, vars, &mut results)?;
        }

        Ok(results)
    }

    fn collect_records<R: std::io::Read>(
        &self,
        reader: &mut csv::Reader<R>,
        vars: &Env<'a>,
        results: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        if *self.headers {
            let headers = reader
                .headers()
                .map_err(|e| format!("Failed to read CSV headers: {e}"))?
                .iter()
                .map(|s| Arc::new(String::from(s)))
                .collect::<Vec<_>>();
            for record in reader.records() {
                let record = record.map_err(|e| format!("Failed to read CSV record: {e}"))?;
                let mut env = vars.clone_pooled(self.runtime.env_pool);
                env.insert(
                    self.var,
                    Value::Map(Arc::new(
                        record
                            .iter()
                            .enumerate()
                            .filter_map(|(i, field)| {
                                if field.is_empty() {
                                    None
                                } else {
                                    Some((
                                        headers
                                            .get(i)
                                            .cloned()
                                            .unwrap_or_else(|| Arc::new(format!("col_{i}"))),
                                        Value::String(Arc::new(String::from(field))),
                                    ))
                                }
                            })
                            .collect::<OrderMap<_, _>>(),
                    )),
                );
                results.push(env);
            }
        } else {
            for record in reader.records() {
                let record = record.map_err(|e| format!("Failed to read CSV record: {e}"))?;
                let mut env = vars.clone_pooled(self.runtime.env_pool);
                env.insert(
                    self.var,
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
                    )),
                );
                results.push(env);
            }
        }
        Ok(())
    }
}

impl<'a> Iterator for LoadCsvOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut envs);

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for vars in batch.active_env_iter() {
                let path = match ExprEval::from_runtime(self.runtime).eval(
                    self.file_path,
                    self.file_path.root().idx(),
                    Some(vars),
                    None,
                ) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                let delimiter = match ExprEval::from_runtime(self.runtime).eval(
                    self.delimiter,
                    self.delimiter.root().idx(),
                    Some(vars),
                    None,
                ) {
                    Ok(Value::String(s)) => s,
                    Ok(_) => return Some(Err(String::from("Delimiter must be a string"))),
                    Err(e) => return Some(Err(e)),
                };
                if delimiter.len() != 1 {
                    return Some(Err(String::from(
                        "CSV field terminator can only be one character wide",
                    )));
                }
                let Value::String(path) = path else {
                    return Some(Err(String::from("File path must be a string")));
                };
                let path = if let Some(path) = path.strip_prefix("file://") {
                    let joined = self.runtime.import_folder.clone() + path;
                    let import_folder = match Path::new(&self.runtime.import_folder).canonicalize()
                    {
                        Ok(p) => p,
                        Err(e) => {
                            return Some(Err(format!(
                                "Failed to canonicalize import folder path '{}': {e}",
                                self.runtime.import_folder
                            )));
                        }
                    };
                    let cpath = match Path::new(&joined).canonicalize() {
                        Ok(p) => p,
                        Err(e) => {
                            return Some(Err(format!(
                                "Failed to canonicalize file path '{joined}': {e}"
                            )));
                        }
                    };
                    if !cpath.starts_with(&import_folder) {
                        return Some(Err(format!(
                            "File path '{joined}' is not within the import folder '{}'",
                            self.runtime.import_folder
                        )));
                    }
                    // Use the canonicalized path for actual I/O so a symlink
                    // race cannot cause us to read a file outside the import
                    // folder between the check and the open.
                    cpath.to_string_lossy().into_owned()
                } else if path.starts_with("https://") {
                    String::from(path.as_str())
                } else {
                    return Some(Err(String::from(
                        "File path must start with 'file://' prefix",
                    )));
                };

                // Read CSV and expand rows
                match self.load_csv_records(&path, &delimiter, vars) {
                    Ok(rows) => {
                        self.pending.extend(rows);
                    }
                    Err(e) => return Some(Err(e)),
                }

                super::drain_pending(&mut self.pending, &mut envs);

                if envs.len() >= BATCH_SIZE {
                    break;
                }
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
