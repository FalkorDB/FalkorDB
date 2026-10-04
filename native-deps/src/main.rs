//! The `native-deps` binary: entry points for work outside `cargo build`.
//!
//! `cargo build` needs none of this -- `graph/build.rs` calls
//! [`native_deps::ensure`] itself. This exists for the Docker dep stages, which
//! build the deps in a layer that holds no FalkorDB workspace (so an apt or pip
//! change cannot invalidate a GraphBLAS build), and for maintenance.

use std::process::ExitCode;

use native_deps::{Dep, Request, Result, ensure, err, lock};

const USAGE: &str = "\
USAGE:
    native-deps [DEP...]       build or reuse the deps (default: all)
    native-deps lock [--check] regenerate / verify deps/native-deps.lock

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
