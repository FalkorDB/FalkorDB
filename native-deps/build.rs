//! Embeds a hash of the recipe sources as `NATIVE_DEPS_RECIPE_HASH`.
//!
//! The cmake flags, patch handling and artifact collection live in Rust, so
//! editing them has to produce new cache keys. Hashing at compile time (rather
//! than reading the sources at runtime) keeps the library usable as a
//! build-dependency from a checkout that may not be where it was compiled.

include!("src/hash.rs");

fn main() {
    println!("cargo:rerun-if-changed=src");

    let src =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("src");
    // Only the code that decides what lands in an artifact: the recipes, the
    // compiler flags, and the helpers that pick which archives get published.
    // The cache, key and CLI code can change without touching a single byte of
    // output, and including them would rebuild all three deps for a comment.
    let mut files = vec![src.join("toolchain.rs"), src.join("util.rs")];
    files.extend(
        collect_files(&src.join("recipes"))
            .expect("list src/recipes")
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "rs")),
    );
    let hash = hash_files(&src, &files).expect("hash the recipe sources");
    println!("cargo:rustc-env=NATIVE_DEPS_RECIPE_HASH={hash}");
}
