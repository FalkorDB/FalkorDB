//! `feral-amd` CLI: read a CSC pattern from a whitespace-separated
//! triplet file (one `row col` pair per line, 0-indexed) and print the
//! AMD permutation plus diagnostic counters.
//!
//! Usage: `feral-amd <triplet.txt>`
//!
//! Triplet format: `#` and `%` comments allowed. The tool
//! symmetrizes the input (unions both halves) and drops duplicates,
//! since AMD requires a full symmetric pattern.

use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::ExitCode;

use feral_amd::{amd_order_with_stats, CscPattern, OrderingError};

fn read_triplet(path: &Path) -> std::io::Result<(usize, Vec<i64>, Vec<i64>)> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut set: BTreeSet<(i64, i64)> = BTreeSet::new();
    let mut max_idx = 0i64;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let r: i64 = parts[0].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad row: {}", e))
        })?;
        let c: i64 = parts[1].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad col: {}", e))
        })?;
        if r < 0 || c < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "negative index in triplet",
            ));
        }
        max_idx = max_idx.max(r).max(c);
        set.insert((r, c));
        set.insert((c, r));
    }
    let n = if set.is_empty() {
        0
    } else {
        (max_idx as usize) + 1
    };
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); n];
    for &(r, c) in &set {
        cols[c as usize].push(r);
    }
    for col in &mut cols {
        col.sort();
    }
    let mut col_ptr: Vec<i64> = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<i64> = Vec::new();
    for col in &cols {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i64);
    }
    Ok((n, col_ptr, row_idx))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: feral-amd <triplet.txt>");
        return ExitCode::from(2);
    }
    let path = Path::new(&args[1]);
    let (n, col_ptr, row_idx) = match read_triplet(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("read error: {}", e);
            return ExitCode::from(2);
        }
    };
    let pattern = match CscPattern::new(n, &col_ptr, &row_idx) {
        Some(p) => p,
        None => {
            eprintln!("pattern failed CSC validation");
            return ExitCode::from(2);
        }
    };
    match amd_order_with_stats(&pattern) {
        Ok((perm, stats)) => {
            println!("n: {}", n);
            println!("nnz: {}", pattern.nnz());
            println!("ncmpa: {}", stats.ncmpa);
            println!("n_dense_deferred: {}", stats.n_dense_deferred);
            println!("ndiv: {}", stats.ndiv);
            println!("nms_ldl: {}", stats.nms_ldl);
            println!("nms_lu: {}", stats.nms_lu);
            print!("perm:");
            for &p in &perm {
                print!(" {}", p);
            }
            println!();
            ExitCode::SUCCESS
        }
        Err(OrderingError::IndexOverflow) => {
            eprintln!("amd: workspace exceeded the index range");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("amd: {}", e);
            ExitCode::from(1)
        }
    }
}
