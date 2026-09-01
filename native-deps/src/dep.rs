//! The three native dependencies this crate knows how to build.

use crate::err;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dep {
    GraphBlas,
    LaGraph,
    RediSearch,
}

impl Dep {
    /// Build order matters: LAGraph links against the GraphBLAS archive.
    pub const ALL: [Self; 3] = [Self::GraphBlas, Self::LaGraph, Self::RediSearch];

    /// The short name used for lock-file sections, cache directories and CLI
    /// arguments.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::GraphBlas => "graphblas",
            Self::LaGraph => "lagraph",
            Self::RediSearch => "redisearch",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "graphblas" => Ok(Self::GraphBlas),
            "lagraph" => Ok(Self::LaGraph),
            "redisearch" => Ok(Self::RediSearch),
            other => Err(err!(
                "unknown dep `{other}` (expected one of: graphblas, lagraph, redisearch)"
            )),
        }
    }
}

impl std::fmt::Display for Dep {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
