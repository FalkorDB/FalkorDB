//! Compiler and OpenMP detection, ported from `graphblas.sh`.

use std::path::PathBuf;

use crate::error::Result;
use crate::util::{env_opt, host_triple, jobs, version_line};

/// How OpenMP gets wired into the cmake builds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenMp {
    /// The Docker toolchain image, where `build/libomp.sh` has produced
    /// `${LIBOMP_PREFIX}/lib/libomp.a`. Forcing GraphBLAS to resolve OpenMP
    /// against that static archive is what makes `libfalkordb.so` embed the
    /// libomp ABI instead of depending on `libgomp.so.1` at runtime.
    Static { prefix: PathBuf },
    /// macOS with Homebrew's libomp. Apple clang has no `-fopenmp` driver flag,
    /// so cmake's `find_package(OpenMP)` fails *silently* and GraphBLAS builds
    /// single-threaded (measured ~4x slower harmonic centrality, ~3x slower MSF
    /// against the C engine). `-Xclang -fopenmp` bypasses the driver check and
    /// works on both Apple clang and Homebrew LLVM clang.
    Brew { prefix: PathBuf },
    /// Everything else: let cmake's `find_package(OpenMP)` auto-detect.
    Auto,
}

impl OpenMp {
    /// The key-visible discriminator. Deliberately excludes the prefix path:
    /// what changes the artifacts is static-vs-dynamic, and the target triple
    /// already distinguishes the Homebrew prefixes that differ by architecture.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Static { .. } => "static",
            Self::Brew { .. } => "brew",
            Self::Auto => "auto",
        }
    }

    /// `-I<prefix>/include`, appended to the shared compiler flags.
    #[must_use]
    pub fn include_flag(&self) -> Option<String> {
        match self {
            Self::Static { prefix } | Self::Brew { prefix } => {
                Some(format!("-I{}/include", prefix.display()))
            }
            Self::Auto => None,
        }
    }

    #[must_use]
    pub fn cmake_args(&self) -> Vec<String> {
        match self {
            Self::Static { prefix } => vec![
                "-DOpenMP_C_FLAGS=-fopenmp=libomp".to_owned(),
                "-DOpenMP_CXX_FLAGS=-fopenmp=libomp".to_owned(),
                "-DOpenMP_C_LIB_NAMES=omp".to_owned(),
                "-DOpenMP_CXX_LIB_NAMES=omp".to_owned(),
                format!("-DOpenMP_omp_LIBRARY={}/lib/libomp.a", prefix.display()),
            ],
            Self::Brew { prefix } => vec![
                format!(
                    "-DOpenMP_C_FLAGS=-Xclang -fopenmp -I{}/include",
                    prefix.display()
                ),
                format!(
                    "-DOpenMP_CXX_FLAGS=-Xclang -fopenmp -I{}/include",
                    prefix.display()
                ),
                "-DOpenMP_C_LIB_NAMES=omp".to_owned(),
                "-DOpenMP_CXX_LIB_NAMES=omp".to_owned(),
                format!("-DOpenMP_omp_LIBRARY={}/lib/libomp.dylib", prefix.display()),
            ],
            Self::Auto => Vec::new(),
        }
    }
}

/// Compiler flags shared by GraphBLAS and LAGraph.
///
/// All in one bucket on purpose: `CMAKE_BUILD_TYPE` is deliberately left unset
/// so cmake never appends a `CMAKE_C_FLAGS_<TYPE>` variant on top -- what you
/// read here is exactly what hits the compile line.
///
/// * `-O3 -g -DNDEBUG` -- RelWithDebInfo-equivalent, except cmake's
///   RelWithDebInfo defaults to `-O2` and we want `-O3`.
/// * `-fPIC` -- both archives get linked into `libfalkordb.{so,dylib}`, a
///   position-independent shared object, so every translation unit must be PIC.
/// * `-fno-stack-protector` -- matches the FalkorDB C engine; drops the stack
///   canary epilogue from hot loops.
/// * `-Wno-incompatible-pointer-types` -- clang-22 promoted this to an error by
///   default and GraphBLAS v10.3.1's `GB_I_inverse.c` trips it.
const COMMON_C_FLAGS: &str =
    "-O3 -g -DNDEBUG -fPIC -fno-stack-protector -Wno-incompatible-pointer-types";

#[derive(Debug, Clone)]
pub struct Toolchain {
    /// `$CC`, if set. Unset means "let cmake pick", matching the shell scripts.
    pub cc: Option<String>,
    pub cxx: Option<String>,
    /// First line of `--version` for the C compiler that will actually be used.
    /// Static archives are ABI-tied to their compiler, and apt.llvm.org moves
    /// clang point releases underneath us, so this belongs in the cache key.
    pub cc_version: String,
    pub cxx_version: String,
    pub target: String,
    pub openmp: OpenMp,
    pub jobs: usize,
}

impl Toolchain {
    pub fn detect() -> Result<Self> {
        let cc = env_opt("CC");
        let cxx = env_opt("CXX");
        let cc_probe = cc.clone().unwrap_or_else(|| "cc".to_owned());
        let cxx_probe = cxx.clone().unwrap_or_else(|| "c++".to_owned());

        Ok(Self {
            cc_version: version_line(&cc_probe).unwrap_or_else(|| format!("unknown ({cc_probe})")),
            cxx_version: version_line(&cxx_probe)
                .unwrap_or_else(|| format!("unknown ({cxx_probe})")),
            cc,
            cxx,
            target: host_triple(),
            openmp: detect_openmp(),
            jobs: jobs(),
        })
    }

    /// `-DCMAKE_C_COMPILER=` / `-DCMAKE_CXX_COMPILER=`, omitted when `$CC`/
    /// `$CXX` are unset so cmake keeps its own detection.
    #[must_use]
    pub fn compiler_cmake_args(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(cc) = &self.cc {
            out.push(format!("-DCMAKE_C_COMPILER={cc}"));
        }
        if let Some(cxx) = &self.cxx {
            out.push(format!("-DCMAKE_CXX_COMPILER={cxx}"));
        }
        out
    }

    #[must_use]
    pub fn common_c_flags(&self) -> String {
        self.openmp.include_flag().map_or_else(
            || COMMON_C_FLAGS.to_owned(),
            |inc| format!("{COMMON_C_FLAGS} {inc}"),
        )
    }
}

fn detect_openmp() -> OpenMp {
    let prefix =
        PathBuf::from(env_opt("LIBOMP_PREFIX").unwrap_or_else(|| "/opt/libomp".to_owned()));
    if prefix.join("lib/libomp.a").is_file() {
        return OpenMp::Static { prefix };
    }
    if cfg!(target_os = "macos")
        && let Some(brew) = crate::util::capture_opt("brew", &["--prefix", "libomp"])
    {
        let brew = PathBuf::from(brew.trim());
        if brew.join("lib/libomp.dylib").is_file() {
            return OpenMp::Brew { prefix: brew };
        }
    }
    OpenMp::Auto
}
