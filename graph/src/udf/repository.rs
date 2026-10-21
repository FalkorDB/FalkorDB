//! # UDF Library Repository
//!
//! This module provides thread-safe, versioned storage for user-defined
//! function libraries. Each library is a named JavaScript source file that
//! registers one or more functions via `falkor.register()`.
//!
//! ## Data Model
//!
//! ```text
//! UdfRepo (process-wide singleton)
//!   |-- version: AtomicU64          // bumped on every mutation
//!   '-- inner: RwLock<Vec<UdfLibrary>>
//!              |
//!              |-- UdfLibrary { name: "mylib", code: "...", function_names: ["mylib.fn1", ...] }
//!              '-- UdfLibrary { name: "utils", code: "...", function_names: ["utils.add", ...] }
//! ```
//!
//! ## Versioning
//!
//! Every mutation (`load`, `delete`, `flush`) increments `version`. Thread-local
//! QuickJS contexts in [`js_context`](super::js_context) compare their cached
//! version against the repo version; on mismatch the context is rebuilt with
//! the latest set of libraries.
//!
//! ## Library Lifecycle
//!
//! 1. **Load** -- Validate the script in a temporary JS context
//!    ([`js_context::validate_script`](super::js_context::validate_script)),
//!    then store the library. Optionally replaces an existing library of the
//!    same name.
//! 2. **Delete** -- Remove a library by name and unregister its functions from
//!    the Cypher function registry.
//! 3. **Flush** -- Remove all libraries at once.
//! 4. **Serialize / Deserialize** -- Support RDB persistence so libraries
//!    survive Redis restarts.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::udf::js_context;

#[derive(Clone, Debug)]
pub struct UdfLibrary {
    pub name: String,
    pub code: String,
    /// Qualified function names registered by this library (e.g., `["lib.func1", "lib.func2"]`).
    pub function_names: Vec<String>,
}

pub struct LibraryInfo {
    pub name: String,
    pub function_names: Vec<String>,
    pub code: Option<String>,
}

pub struct UdfRepo {
    inner: RwLock<UdfRepoInner>,
    version: AtomicU64,
}

struct UdfRepoInner {
    /// Libraries stored in insertion order.
    libraries: Vec<UdfLibrary>,
}

impl Default for UdfRepo {
    fn default() -> Self {
        Self::new()
    }
}

impl UdfRepo {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: RwLock::new(UdfRepoInner {
                libraries: Vec::new(),
            }),
            version: AtomicU64::new(0),
        }
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Bump the version counter to force thread-local JS context rebuilds.
    /// Used when JS config (heap size, stack size) changes.
    pub fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Load or replace a library. Returns the list of qualified function names.
    pub fn load(
        &self,
        name: &str,
        code: &str,
        replace: bool,
    ) -> Result<Vec<String>, String> {
        // Phase 1: Validate the script in a temporary JS context and collect registered names
        let raw_names = js_context::validate_script(code)?;

        let qualified_names: Vec<String> = raw_names
            .iter()
            .map(|fn_name| format!("{name}.{fn_name}"))
            .collect();

        // Phase 2: Store in the repository
        let mut inner = self.inner.write();
        let existing_pos = inner.libraries.iter().position(|lib| lib.name == name);

        if !replace && existing_pos.is_some() {
            return Err(format!(
                "Failed to register UDF library '{name}': library already registered"
            ));
        }

        // If replacing, remove old library first
        if let Some(pos) = existing_pos {
            let old_lib = inner.libraries.remove(pos);
            for old_fn in &old_lib.function_names {
                crate::runtime::functions::unregister_udf(old_fn);
            }
        }

        let lib = UdfLibrary {
            name: name.to_string(),
            code: code.to_string(),
            function_names: qualified_names.clone(),
        };
        inner.libraries.push(lib);

        // Phase 3: Bump version
        self.version.fetch_add(1, Ordering::Release);

        Ok(qualified_names)
    }

    /// Delete a library by name. Returns the list of function names that were removed.
    pub fn delete(
        &self,
        name: &str,
    ) -> Result<Vec<String>, String> {
        let mut inner = self.inner.write();
        let pos = inner
            .libraries
            .iter()
            .position(|lib| lib.name == name)
            .ok_or_else(|| format!("Library '{name}' does not exist"))?;
        let lib = inner.libraries.remove(pos);

        self.version.fetch_add(1, Ordering::Release);
        Ok(lib.function_names)
    }

    /// Remove all libraries. Returns all function names that were removed.
    pub fn flush(&self) -> Vec<String> {
        let mut inner = self.inner.write();
        let all_names: Vec<String> = inner
            .libraries
            .iter()
            .flat_map(|lib| lib.function_names.clone())
            .collect();
        inner.libraries.clear();
        self.version.fetch_add(1, Ordering::Release);
        all_names
    }

    /// List loaded libraries, optionally filtered by name.
    pub fn list(
        &self,
        filter: Option<&str>,
        with_code: bool,
    ) -> Vec<LibraryInfo> {
        let inner = self.inner.read();
        inner
            .libraries
            .iter()
            .filter(|lib| filter.is_none_or(|f| lib.name == f))
            .map(|lib| {
                // Extract just the function names (without the "lib." prefix)
                let fn_names = lib
                    .function_names
                    .iter()
                    .map(|qn| {
                        qn.strip_prefix(&format!("{}.", lib.name))
                            .unwrap_or(qn)
                            .to_string()
                    })
                    .collect();
                LibraryInfo {
                    name: lib.name.clone(),
                    function_names: fn_names,
                    code: if with_code {
                        Some(lib.code.clone())
                    } else {
                        None
                    },
                }
            })
            .collect()
    }

    /// Get all libraries for context rebuild, in insertion order.
    pub fn get_all_libraries(&self) -> Vec<UdfLibrary> {
        let inner = self.inner.read();
        inner.libraries.clone()
    }

    /// Serialize all libraries for persistence.
    pub fn serialize(&self) -> Vec<(String, String)> {
        let inner = self.inner.read();
        inner
            .libraries
            .iter()
            .map(|lib| (lib.name.clone(), lib.code.clone()))
            .collect()
    }

    /// Bulk load from deserialized data (for RDB load).
    ///
    /// Validates **all** libraries in a staging area first. Only on full
    /// success are the live repo contents atomically replaced and the
    /// version bumped. On any validation failure the live repo and
    /// runtime function table remain unchanged.
    pub fn deserialize(
        &self,
        libs: &[(String, String)],
    ) -> Result<Vec<UdfLibrary>, String> {
        // Phase 1 – validate every library and build the staging list.
        let mut staged: Vec<UdfLibrary> = Vec::with_capacity(libs.len());
        for (name, code) in libs {
            let raw_names = js_context::validate_script(code).map_err(|e| {
                format!("UDF library '{name}' failed validation during RDB load: {e}")
            })?;
            let qualified_names: Vec<String> = raw_names
                .iter()
                .map(|fn_name| format!("{name}.{fn_name}"))
                .collect();
            staged.push(UdfLibrary {
                name: name.clone(),
                code: code.clone(),
                function_names: qualified_names,
            });
        }

        // Phase 2 – all validated; atomically swap the live repo.
        let result = staged.clone();
        {
            let mut inner = self.inner.write();

            // Unregister old functions.
            for lib in &inner.libraries {
                for qname in &lib.function_names {
                    crate::runtime::functions::unregister_udf(qname);
                }
            }

            inner.libraries = staged;
        }
        self.version.fetch_add(1, Ordering::Release);

        Ok(result)
    }
}
