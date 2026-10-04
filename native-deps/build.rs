//! Embeds a hash of the recipe sources as `NATIVE_DEPS_RECIPE_HASH`.
//!
//! The cmake flags, patch handling and artifact collection live in Rust, so
//! editing them has to produce new cache keys. Hashing at compile time (rather
//! than reading the sources at runtime) keeps the library usable as a
//! build-dependency from a checkout that may not be where it was compiled.

include!("src/sha256.rs");

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src");

    let src =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("src");
    // Only the code that decides what lands in an artifact: the recipes, the
    // compiler flags, and the helpers that pick which archives get published.
    // The cache, key and CLI code can change without touching a single byte of
    // output, and including them would rebuild all three deps for a comment.
    let mut files = vec![src.join("toolchain.rs"), src.join("util.rs")];
    collect(&src.join("recipes"), &mut files);
    files.sort();

    let mut hasher = Sha256::new();
    for path in &files {
        let rel = path.strip_prefix(&src).unwrap_or(path);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(&fs::read(path).unwrap_or_default());
        hasher.update(b"\n");
    }

    println!(
        "cargo:rustc-env=NATIVE_DEPS_RECIPE_HASH={}",
        hex(&hasher.finish())
    );
}

fn collect(
    dir: &Path,
    out: &mut Vec<PathBuf>,
) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
