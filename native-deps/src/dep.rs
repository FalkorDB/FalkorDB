//! The native dependencies this crate knows how to build.

use crate::err;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dep {
    GraphBlas,
    LaGraph,
    GkLib,
    Metis,
    RediSearch,
}

impl Dep {
    /// Build order matters: LAGraph links against the GraphBLAS archive, and
    /// METIS is compiled against GKlib's headers.
    pub const ALL: [Self; 5] = [
        Self::GraphBlas,
        Self::LaGraph,
        Self::GkLib,
        Self::Metis,
        Self::RediSearch,
    ];

    /// The short name used for lock-file sections, cache directories and CLI
    /// arguments.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::GraphBlas => "graphblas",
            Self::LaGraph => "lagraph",
            Self::GkLib => "gklib",
            Self::Metis => "metis",
            Self::RediSearch => "redisearch",
        }
    }

    /// The dep whose build this one consumes, and whose key is therefore part
    /// of this one's: LAGraph links that exact GraphBLAS archive, METIS is
    /// compiled against that exact GKlib's headers. It is always resolved
    /// first ([`Self::ALL`] lists it earlier).
    #[must_use]
    pub const fn upstream(self) -> Option<Self> {
        match self {
            Self::LaGraph => Some(Self::GraphBlas),
            Self::Metis => Some(Self::GkLib),
            Self::GraphBlas | Self::GkLib | Self::RediSearch => None,
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "graphblas" => Ok(Self::GraphBlas),
            "lagraph" => Ok(Self::LaGraph),
            "gklib" => Ok(Self::GkLib),
            "metis" => Ok(Self::Metis),
            "redisearch" => Ok(Self::RediSearch),
            other => Err(err!(
                "unknown dep `{other}` (expected one of: graphblas, lagraph, gklib, metis, \
                 redisearch)"
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

#[cfg(test)]
mod tests {
    use super::Dep;

    #[test]
    fn names_round_trip_through_parse_and_display() {
        for dep in Dep::ALL {
            assert_eq!(Dep::parse(dep.name()).unwrap(), dep);
            assert_eq!(dep.to_string(), dep.name());
        }
        assert_eq!(Dep::parse(" GraphBLAS ").unwrap(), Dep::GraphBlas);
        assert!(Dep::parse("graphblas2").is_err());
    }

    #[test]
    fn an_upstream_is_resolved_before_its_dependent() {
        for (i, dep) in Dep::ALL.iter().enumerate() {
            if let Some(up) = dep.upstream() {
                assert!(Dep::ALL[..i].contains(&up), "{dep} is listed before {up}");
            }
        }
    }
}
