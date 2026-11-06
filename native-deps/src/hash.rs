// SHA-256 helpers, shared with build.rs, which `include!`s this file to compute
// the recipe hash. That is why it uses `//` comments (an included file cannot
// carry `//!` inner docs), only `std` and `sha2`, and `io::Result` rather than
// the crate's error type.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Streaming SHA-256 of a file's contents.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// One SHA-256 over `files`, hashing each one's path relative to `base`
/// alongside its contents, so a rename is a change but moving `base` is not.
pub fn hash_files(
    base: &Path,
    files: &[PathBuf],
) -> io::Result<String> {
    let mut files = files.to_vec();
    files.sort();
    let mut hasher = Sha256::new();
    for path in &files {
        let rel = path.strip_prefix(base).unwrap_or(path);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(hash_file(path)?.as_bytes());
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// [`hash_files`] over every file under `dir` that `keep` accepts. A missing
/// directory hashes as empty: `build/graphblas/PreJIT` is legitimately absent
/// before the first harvest.
pub fn hash_tree(
    dir: &Path,
    keep: &dyn Fn(&Path) -> bool,
) -> io::Result<String> {
    let mut files = collect_files(dir)?;
    files.retain(|p| keep(p));
    hash_files(dir, &files)
}

/// Every file under `dir`, recursively; empty if `dir` does not exist.
pub fn collect_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                stack.push(entry.path());
            } else {
                out.push(entry.path());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
