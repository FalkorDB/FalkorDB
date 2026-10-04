//! The `native-deps` binary: entry points for work outside `cargo build`.
//!
//! `cargo build` needs none of this -- `graph/build.rs` calls
//! [`native_deps::ensure`] itself. This exists for the Docker dep stages, which
//! build the deps in a layer that holds no FalkorDB workspace (so an apt or pip
//! change cannot invalidate a GraphBLAS build), and for maintenance.

use std::process::ExitCode;
use std::time::Duration;

use native_deps::cache::Cache;
use native_deps::prune::{self, Options};
use native_deps::{Dep, Request, Result, current_entries, ensure, err, lock};

const USAGE: &str = "\
USAGE:
    native-deps [DEP...]       build or reuse the deps (default: all)
    native-deps lock [--check] regenerate / verify deps/native-deps.lock
    native-deps prune [--dry-run] [--days N] [--worktree]
                               remove cache entries this checkout does not use
                               and nothing used for N days (default 14); and
                               with --worktree, deps/RediSearch/bin too

DEP is graphblas, lagraph or redisearch. The flavour comes from the same
environment `cargo build` reads: REDISEARCH_SAN, FALKORDB_PREJIT_HARVEST,
FALKORDB_NATIVE_DEPS_FORCE, CC/CXX, FALKORDB_DEPS_CACHE, ...
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("native-deps: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h" | "--help" | "help") => print!("{USAGE}"),
        Some("lock") => {
            let root = lock::root_for(&std::env::current_dir()?)?;
            match args.get(1).map(String::as_str) {
                None => println!("{}", lock::write(&root)?),
                Some("--check") => {
                    lock::check(&root)?;
                    println!("{} is up to date", lock::LOCK_RELPATH);
                }
                Some(other) => return Err(err!("unknown option `{other}`\n\n{USAGE}")),
            }
        }
        Some("prune") => cmd_prune(&args[1..])?,
        _ => {
            let mut request = Request::from_env()?;
            if !args.is_empty() {
                request.deps = args.iter().map(|a| Dep::parse(a)).collect::<Result<_>>()?;
            }
            for (dep, resolved) in ensure(&request)?.iter() {
                println!("{dep}\t{}\t{}", resolved.key, resolved.prefix.display());
            }
        }
    }
    Ok(())
}

fn cmd_prune(args: &[String]) -> Result<()> {
    let request = Request::from_env()?;
    let mut opts = Options {
        keep_for: Duration::from_secs(14 * 24 * 3600),
        keep_keys: current_entries(&request)?,
        worktree: None,
        dry_run: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dry-run" => opts.dry_run = true,
            "--worktree" => opts.worktree = Some(request.root.clone()),
            "--days" => {
                let days: u64 = it
                    .next()
                    .and_then(|d| d.parse().ok())
                    .ok_or_else(|| err!("--days needs a number"))?;
                opts.keep_for = Duration::from_secs(days * 24 * 3600);
            }
            other => return Err(err!("unknown option `{other}`\n\n{USAGE}")),
        }
    }

    let removals = prune::prune(&Cache::discover()?, &opts)?;
    let verb = if opts.dry_run {
        "would remove"
    } else {
        "removed"
    };
    let mut total = 0;
    for r in &removals {
        total += r.bytes;
        println!(
            "{verb} {} ({}, {})",
            r.path.display(),
            mib(r.bytes),
            r.reason
        );
    }
    println!("{verb} {} item(s), {}", removals.len(), mib(total));
    Ok(())
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}
