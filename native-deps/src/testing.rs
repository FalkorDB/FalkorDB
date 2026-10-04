//! Helpers shared by the unit tests.

use std::fs;
use std::path::PathBuf;

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "native-deps-{name}-{}-{}",
            std::process::id(),
            crate::util::now_secs()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
