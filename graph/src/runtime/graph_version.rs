//! The graph version a query runs on.
//!
//! A read query runs on a published version, an `Arc<Graph>` nothing can
//! change. A write query runs on its fork, a `Graph` it owns
//! ([`MvccGraph::fork`]). Operators reach the runtime through shared
//! references, so the fork sits in a `RefCell`, and the runtime publishes it by
//! moving it out ([`GraphVersion::publish`]); from then on the query reads the
//! version it published. The query, and with it the fork, stays on one thread.
//!
//! [`MvccGraph::fork`]: crate::graph::mvcc_graph::MvccGraph::fork

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use crate::graph::graph::Graph;

/// A version a query runs on. Cloning it shares the version: the JS UDF
/// context keeps one for the callbacks it hands to JavaScript.
#[derive(Clone)]
pub enum GraphVersion {
    /// A read query's published version.
    Published(Arc<Graph>),
    /// A write query's version: its fork, until the fork is published.
    Write(Rc<RefCell<Write>>),
}

/// A write query's version.
// One per write query, behind its `Rc`: boxing the fork would only add an allocation.
#[allow(clippy::large_enum_variant)]
pub enum Write {
    /// The fork the query builds into.
    Building(Graph),
    /// The fork, published.
    Published(Arc<Graph>),
    /// [`MvccGraph::commit`](crate::graph::mvcc_graph::MvccGraph::commit)
    /// refused the fork, which is gone with it.
    Refused,
}

impl GraphVersion {
    /// Shared access to the version.
    ///
    /// # Panics
    ///
    /// If the version is a fork the query is changing, or a refused one.
    #[must_use]
    pub fn borrow(&self) -> GraphRef<'_> {
        match self {
            Self::Published(g) => GraphRef::Published(g),
            Self::Write(w) => GraphRef::Write(Ref::map(w.borrow(), |w| match w {
                Write::Building(g) => g,
                Write::Published(g) => &**g,
                Write::Refused => panic!("the version was refused at commit"),
            })),
        }
    }

    /// Exclusive access to a fork that is not published yet.
    ///
    /// # Panics
    ///
    /// On a published version, or if the fork is borrowed.
    #[must_use]
    pub fn borrow_mut(&self) -> RefMut<'_, Graph> {
        let Self::Write(w) = self else {
            panic!("a read query's version is published, and never changed");
        };
        RefMut::map(w.borrow_mut(), |w| match w {
            Write::Building(g) => g,
            Write::Published(_) | Write::Refused => {
                panic!("a write query's version is changed only until it is committed")
            }
        })
    }

    /// Publish the fork with `commit`, which takes it by value and returns
    /// the version it published; from then on this reads that version.
    ///
    /// # Errors
    ///
    /// Whatever `commit` returns. The fork is gone then: `commit` consumed it.
    ///
    /// # Panics
    ///
    /// On a published version, a fork published or refused already, or a fork
    /// that is borrowed.
    pub fn publish<E>(
        &self,
        commit: impl FnOnce(Graph) -> Result<Arc<Graph>, E>,
    ) -> Result<(), E> {
        let Self::Write(w) = self else {
            panic!("a read query has no fork to publish");
        };
        let mut w = w.borrow_mut();
        let Write::Building(fork) = std::mem::replace(&mut *w, Write::Refused) else {
            panic!("a write query's fork is published once");
        };
        *w = Write::Published(commit(fork)?);
        Ok(())
    }
}

impl From<Arc<Graph>> for GraphVersion {
    fn from(g: Arc<Graph>) -> Self {
        Self::Published(g)
    }
}

impl From<Graph> for GraphVersion {
    fn from(fork: Graph) -> Self {
        Self::Write(Rc::new(RefCell::new(Write::Building(fork))))
    }
}

/// Shared access to a [`GraphVersion`].
pub enum GraphRef<'r> {
    /// A published version needs no borrow: nothing can change it.
    Published(&'r Graph),
    /// A write query's borrow, which keeps the fork from being changed meanwhile.
    Write(Ref<'r, Graph>),
}

impl Deref for GraphRef<'_> {
    type Target = Graph;

    fn deref(&self) -> &Graph {
        match self {
            Self::Published(g) => g,
            Self::Write(g) => g,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphRef, GraphVersion};
    use crate::graph::{graphblas::test_init::ensure_init, mvcc_graph::MvccGraph};

    /// A write query's version is its fork until it is published, and the
    /// published version from then on: the query reads what `MvccGraph`
    /// published, and can no longer change it.
    #[test]
    fn a_published_fork_reads_the_published_version_and_stays_unchanged() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let version = GraphVersion::from(mvcc.fork());
        version.borrow_mut().get_label_id_mut("L");

        version
            .publish(|fork| mvcc.commit(fork))
            .expect("a new label commits");

        let GraphRef::Write(read) = version.borrow() else {
            panic!("a write query's version");
        };
        assert!(std::ptr::eq(&*read, &*mvcc.read()), "the published version");
        assert_eq!(read.get_labels().len(), 1);
        drop(read);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(version.borrow_mut());
            }))
            .is_err(),
            "a published version must not be changed"
        );
    }

    /// A read query's version is the published one, and is never changed.
    #[test]
    #[should_panic(expected = "a read query's version is published")]
    fn a_read_querys_version_is_never_changed() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let version = GraphVersion::from(mvcc.read());
        assert!(
            matches!(version.borrow(), GraphRef::Published(g) if std::ptr::eq(g, &*mvcc.read()))
        );
        drop(version.borrow_mut());
    }
}
