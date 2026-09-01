//! `native-deps` CLI.
//!
//! Same code as the library, exposed for the Docker dep stages (which must build
//! a dependency before any Rust source is in the image), for CI cache keys, and
//! for developers who want to drive a build by hand.

use std::fs;
use std::process::ExitCode;

use native_deps::key::RECIPE_HASH;
use native_deps::lock::{self, LOCK_RELPATH, LockFile};
use native_deps::util::{env_flag, env_opt, find_repo_root};
use native_deps::{Dep, Request, Resolution, Result, ensure, err, keys};

const USAGE: &str = "\
native-deps -- build and cache FalkorDB's native dependencies

USAGE:
    native-deps ensure [OPTIONS]        resolve every dep, building on a miss
    native-deps build  <dep>... [OPTS]  build specific deps (used by Docker stages)
    native-deps key    [OPTIONS]        print cache keys (used by CI cache keys)
    native-deps lock   [--check]        regenerate / verify deps/native-deps.lock

DEPS:
    graphblas  lagraph  redisearch

OPTIONS:
    --dep <name>     restrict to a dep; repeatable (default: all)
    --san <kind>     sanitizer flavor for RediSearch, e.g. `address`
    --force          rebuild even on a cache hit
    --offline        fail instead of building on a cache miss
    --combined       (key) print one hash covering every selected dep
    --json           machine-readable output
    --root <dir>     FalkorDB checkout to operate on (default: discovered)
    -h, --help       this text

ENVIRONMENT:
    FALKORDB_DEPS_CACHE             cache root (default $HOME/.cache/falkordb/native-deps)
    FALKORDB_NATIVE_DEPS_PREBUILT   colon-separated read-only cache roots, searched first
    FALKORDB_NATIVE_DEPS_OFFLINE    same as --offline
    FALKORDB_NATIVE_DEPS_FORCE      same as --force
    FALKORDB_NATIVE_DEPS_KEEP_BUILD keep the scratch cmake tree for debugging
    FALKORDB_PREJIT_HARVEST         build a GraphBLAS with no PreJIT kernels
    FALKORDB_REPO_ROOT              same as --root
    REDISEARCH_SAN                  same as --san
    CC / CXX / LIBOMP_PREFIX / JOBS as for the cmake builds
    GRAPHBLAS_PREFIX, LAGRAPH_PREFIX, REDISEARCH_PREFIX
                                    bypass the cache with a prebuilt prefix
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

#[derive(Default)]
struct Args {
    deps: Vec<Dep>,
    san: Option<String>,
    force: bool,
    offline: bool,
    combined: bool,
    json: bool,
    check: bool,
    root: Option<String>,
}

fn run() -> Result<()> {
    let mut argv = std::env::args().skip(1);
    let Some(command) = argv.next() else {
        print!("{USAGE}");
        return Ok(());
    };
    if command == "-h" || command == "--help" || command == "help" {
        print!("{USAGE}");
        return Ok(());
    }

    let mut args = Args::default();
    let rest: Vec<String> = argv.collect();
    let mut it = rest.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--dep" => args.deps.push(Dep::parse(
                &it.next().ok_or_else(|| err!("--dep needs a value"))?,
            )?),
            "--san" => args.san = Some(it.next().ok_or_else(|| err!("--san needs a value"))?),
            "--root" => args.root = Some(it.next().ok_or_else(|| err!("--root needs a value"))?),
            "--force" => args.force = true,
            "--offline" => args.offline = true,
            "--combined" => args.combined = true,
            "--json" => args.json = true,
            "--check" => args.check = true,
            "--all" => args.deps = Dep::ALL.to_vec(),
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other if other.starts_with('-') => return Err(err!("unknown option `{other}`")),
            // Bare words are dep names, so `native-deps build graphblas` works.
            other => args.deps.push(Dep::parse(other)?),
        }
    }

    match command.as_str() {
        "ensure" | "build" => cmd_ensure(&args),
        "key" => cmd_key(&args),
        "lock" => cmd_lock(&args),
        other => Err(err!("unknown command `{other}`\n\n{USAGE}")),
    }
}

fn request(args: &Args) -> Result<Request> {
    let root = match &args.root {
        Some(dir) => std::path::PathBuf::from(dir),
        None => find_repo_root(&std::env::current_dir()?)?,
    };
    let deps = if args.deps.is_empty() {
        Dep::ALL.to_vec()
    } else {
        args.deps.clone()
    };
    Ok(Request {
        root,
        deps,
        san: args.san.clone().or_else(|| env_opt("REDISEARCH_SAN")),
        prejit_harvest: env_flag("FALKORDB_PREJIT_HARVEST"),
        force: args.force || env_flag("FALKORDB_NATIVE_DEPS_FORCE"),
        offline: args.offline || env_flag("FALKORDB_NATIVE_DEPS_OFFLINE"),
    })
}

fn cmd_ensure(args: &Args) -> Result<()> {
    let resolution = ensure(&request(args)?)?;
    if args.json {
        println!("{}", render_json(&resolution));
    } else {
        for (dep, r) in resolution.iter() {
            println!("{dep}\t{}\t{}", r.key, r.prefix.display());
        }
    }
    Ok(())
}

fn cmd_key(args: &Args) -> Result<()> {
    let req = request(args)?;
    let keys = keys(&req)?;

    if args.combined {
        // One hash over every selected dep, for a single CI cache key.
        let joined: String = keys.iter().map(|(d, k)| format!("{d}={k}\n")).collect();
        let mut digest = native_deps::sha256::sha256_hex(joined.as_bytes());
        digest.truncate(16);
        println!("{digest}");
        return Ok(());
    }

    if args.json {
        let body: Vec<String> = keys
            .iter()
            .map(|(d, k)| format!("  \"{d}\": \"{}\"", escape(k)))
            .collect();
        println!("{{\n{}\n}}", body.join(",\n"));
    } else {
        for (dep, key) in &keys {
            println!("{dep}\t{key}");
        }
    }
    Ok(())
}

fn cmd_lock(args: &Args) -> Result<()> {
    let root = match &args.root {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            // `lock` is the one command that can run before the lock file
            // exists, so fall back to the git toplevel.
            let cwd = std::env::current_dir()?;
            find_repo_root(&cwd).or_else(|e| {
                native_deps::util::capture("git", &["rev-parse", "--show-toplevel"], Some(&cwd))
                    .map(|t| std::path::PathBuf::from(t.trim()))
                    .map_err(|_| e)
            })?
        }
    };

    let generated = lock::from_git(&root)?.render();
    let path = root.join(LOCK_RELPATH);

    if args.check {
        let current =
            fs::read_to_string(&path).map_err(|e| err!("cannot read {}: {e}", path.display()))?;
        if current == generated {
            println!("{LOCK_RELPATH} is up to date");
            return Ok(());
        }
        return Err(err!(
            "{LOCK_RELPATH} is stale.\n\
             A gitlink under deps/ moved without the lock file being regenerated.\n\
             Run `native-deps lock` and commit the result.\n\n\
             expected:\n{}\n\ngot:\n{}",
            summarize(&generated),
            summarize(&current)
        ));
    }

    fs::write(&path, &generated).map_err(|e| err!("cannot write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    let lock = LockFile::parse(&generated, &path)?;
    for e in &lock.entries {
        println!("  {} {} ({})", e.name, &e.rev[..e.rev.len().min(12)], e.pin);
    }
    Ok(())
}

/// Condense a lock file to its `name rev` pairs for a diffable error message.
fn summarize(text: &str) -> String {
    let mut out = Vec::new();
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.to_owned();
        } else if let Some(rev) = line.strip_prefix("rev = ") {
            out.push(format!("  {section} {rev}"));
        }
    }
    out.join("\n")
}

fn render_json(resolution: &Resolution) -> String {
    let deps: Vec<String> = resolution
        .iter()
        .map(|(dep, r)| {
            format!(
                "    \"{dep}\": {{\n      \
                 \"key\": \"{}\",\n      \
                 \"prefix\": \"{}\",\n      \
                 \"include\": \"{}\",\n      \
                 \"lib\": \"{}\"\n    }}",
                escape(&r.key),
                escape(&r.prefix.display().to_string()),
                escape(&r.include().display().to_string()),
                escape(&r.lib().display().to_string()),
            )
        })
        .collect();
    format!(
        "{{\n  \"recipe\": \"{RECIPE_HASH}\",\n  \"deps\": {{\n{}\n  }}\n}}",
        deps.join(",\n")
    )
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
