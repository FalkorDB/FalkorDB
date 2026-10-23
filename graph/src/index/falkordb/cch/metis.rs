//! The C METIS binding behind [`MetisOrdering`](super::order::MetisOrdering).
//!
//! METIS 5 (KarypisLab, with its support library GKlib) is the library the C engine orders
//! with, at the same pins (`deps/KarypisLab/{METIS,GKlib}`), built by `native-deps` with a
//! 64-bit `idx_t` like the C engine's. CCH makes one call, `METIS_NodeND`, with METIS's
//! default options. Three things about that call shape this file:
//!
//! * **The skeleton is copied once**, into `i64` arrays: METIS reads `idx_t`, signed, where
//!   the skeleton's ids are `u64`. The copy also leaves out the isolated nodes and
//!   renumbers the rest ([`metis_graph`]). METIS indexes its input without any check, so
//!   the copy is also where every value it will index with is bounds-checked.
//! * **One call at a time, process-wide** ([`METIS_LOCK`]). METIS seeds GKlib's random
//!   number generator on every call (`InitRandom` in `METIS/libmetis/util.c`) and draws
//!   from it throughout. `native-deps` builds GKlib with `USE_GKRAND`, so that generator
//!   is GKlib's own (`static uint64_t mt[]` in `GKlib/src/random.c`), touched by nothing
//!   but METIS, rather than the C library's `rand`, which the whole process shares. It is
//!   still one state for the process: two overlapping calls would reseed and draw from it
//!   together, and each permutation would depend on the interleaving. The signal-handler
//!   guard below saves and restores process-wide state too, and two overlapping guards
//!   would restore each other's handlers.
//! * **METIS rewrites two process-wide signal handlers on every call**; see
//!   [`preserving_signal_handlers`].

use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::ptr::null_mut;
use std::sync::{Mutex, PoisonError};

use super::Csr;
use super::order::{NONE, OrderError};

// METIS's `idx_t` is `i64`: `native-deps` builds it with `IDXTYPEWIDTH=64`, and
// `graph/build.rs` reads the width back from the `metis.h` installed next to the archive
// actually linked, so a METIS of another width (an override prefix, say) fails here rather
// than reading every array at the wrong stride.
const _: () = assert!(
    matches!(env!("FALKORDB_METIS_IDXTYPEWIDTH").as_bytes(), b"64"),
    "METIS must be built with a 64-bit idx_t"
);

/// `METIS_NOPTIONS` in `metis.h`: the length of the options array.
const METIS_NOPTIONS: usize = 40;
/// `rstatus_et` in `metis.h`.
const METIS_OK: c_int = 1;
const METIS_ERROR_INPUT: c_int = -2;
const METIS_ERROR_MEMORY: c_int = -3;

unsafe extern "C" {
    fn METIS_SetDefaultOptions(options: *mut i64) -> c_int;
    fn METIS_NodeND(
        nvtxs: *mut i64,
        xadj: *mut i64,
        adjncy: *mut i64,
        vwgt: *mut i64,
        options: *mut i64,
        perm: *mut i64,
        iperm: *mut i64,
    ) -> c_int;
}

/// Serialises every METIS call in the process (see the module docs).
static METIS_LOCK: Mutex<()> = Mutex::new(());

/// The skeleton as METIS reads it, `xadj` and `adjncy`, without its isolated nodes.
/// Transient: built just before the call, freed just after.
pub(super) struct MetisGraph {
    xadj: Vec<i64>,
    adjncy: Vec<i64>,
    /// `nodes[k]` is the skeleton node METIS knows as `k`, ascending.
    nodes: Vec<u64>,
    /// The skeleton nodes with no neighbour, ascending. They take the lowest ranks.
    isolated: Vec<u64>,
}

/// Copy `skeleton` into METIS's arrays.
///
/// The skeleton must be the adjacency of a simple undirected graph
/// ([`Csr::is_symmetric_simple`]): what `CchInput::build` produces by construction, and
/// asserted here in debug builds only. Release builds check, in one pass and without
/// allocating, what keeps METIS within its arrays: the offsets ([`Csr::is_well_formed`]),
/// and that every neighbour has a row of its own, so its METIS id exists.
///
/// Isolated nodes (empty rows) are left out and the rest renumbered in id order. They
/// create no fill wherever they are eliminated, so their ranks are free, and a skeleton
/// sized by `node_cap` carries every unused id slot as one: 294k of California's 1.54M
/// nodes, on which METIS would otherwise spend 6% of its instructions and 80 MiB of peak
/// memory.
///
/// The copy costs `8 (m + 1) + 8 nnz` bytes for the `m` nodes METIS sees, plus `8 n` for
/// the renumbering, while METIS runs; it is freed before the permutation is returned.
pub(super) fn metis_graph(skeleton: &Csr) -> Result<MetisGraph, OrderError> {
    if !skeleton.is_well_formed() {
        return Err(OrderError::Backend("malformed skeleton"));
    }
    debug_assert!(
        skeleton.is_symmetric_simple(),
        "the skeleton must be symmetric, sorted, without repeats or self-loops"
    );
    let n = skeleton.n;

    // `renumber[u]` is `u`'s METIS id, or `NONE` for an isolated node.
    let mut renumber = vec![NONE; n];
    let mut nodes = Vec::new();
    let mut isolated = Vec::new();
    for (u, id) in renumber.iter_mut().enumerate() {
        if skeleton.row(u).is_empty() {
            isolated.push(u as u64);
        } else {
            *id = nodes.len() as u64;
            nodes.push(u as u64);
        }
    }

    // CSR row pointers: `m + 1` of them, starting at 0.
    let mut xadj = Vec::with_capacity(nodes.len() + 1);
    let mut adjncy = Vec::with_capacity(skeleton.i.len());
    xadj.push(0);
    for &u in &nodes {
        for &v in skeleton.row(u as usize) {
            // `NONE` is a neighbour with an empty row, which only an asymmetric skeleton
            // has: METIS would read it as -1.
            let id = renumber[v as usize];
            if id == NONE {
                return Err(OrderError::Backend("skeleton is not symmetric"));
            }
            // Below `n <= MAX_NODES < 2^60`, so it fits an `i64`.
            adjncy.push(id as i64);
        }
        xadj.push(adjncy.len() as i64);
    }
    Ok(MetisGraph {
        xadj,
        adjncy,
        nodes,
        isolated,
    })
}

/// `METIS_NodeND` with default options: `perm[rank] = node`, the same "new-to-old"
/// convention as [`Ordering`](super::order::Ordering), with the isolated nodes first.
/// Deterministic: the default seed is fixed, the generator is METIS's alone (`USE_GKRAND`,
/// see the module docs), and [`METIS_LOCK`] keeps other METIS calls off it.
pub(super) fn node_nd(graph: MetisGraph) -> Result<Vec<u64>, OrderError> {
    let _one_at_a_time = METIS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    preserving_signal_handlers(|| node_nd_unguarded(graph))
}

/// The call itself, with no signal-handler guard. Callers hold [`METIS_LOCK`].
fn node_nd_unguarded(graph: MetisGraph) -> Result<Vec<u64>, OrderError> {
    let MetisGraph {
        mut xadj,
        mut adjncy,
        nodes,
        isolated: mut perm_out,
    } = graph;
    let n = nodes.len();
    if n == 0 {
        return Ok(perm_out);
    }
    let mut nvtxs = i64::try_from(n).map_err(|_| OrderError::TooLarge)?;
    let mut options = [0i64; METIS_NOPTIONS];
    let mut perm: Vec<i64> = vec![0; n];
    let mut iperm: Vec<i64> = vec![0; n];

    // SAFETY: `options` holds the METIS_NOPTIONS entries both calls read or write; `xadj`
    // holds `nvtxs + 1` offsets into `adjncy`, each id below `nvtxs` (`metis_graph`
    // checked all of it); `perm` and `iperm` hold `nvtxs` entries each; a null `vwgt`
    // means unit weights. METIS writes only `perm`, `iperm` and `options`: the signature
    // is non-const, but it leaves `xadj` and `adjncy` as it found them (and we own them
    // anyway). Nothing here outlives the call.
    let status = unsafe {
        METIS_SetDefaultOptions(options.as_mut_ptr());
        METIS_NodeND(
            &raw mut nvtxs,
            xadj.as_mut_ptr(),
            adjncy.as_mut_ptr(),
            null_mut(),
            options.as_mut_ptr(),
            perm.as_mut_ptr(),
            iperm.as_mut_ptr(),
        )
    };
    drop((xadj, adjncy, iperm));
    match status {
        METIS_OK => {}
        METIS_ERROR_INPUT => return Err(OrderError::Backend("METIS rejected the skeleton")),
        METIS_ERROR_MEMORY => return Err(OrderError::Backend("METIS ran out of memory")),
        _ => return Err(OrderError::Backend("METIS failed")),
    }
    perm_out.reserve(n);
    for v in perm {
        let node = usize::try_from(v)
            .ok()
            .and_then(|v| nodes.get(v))
            .ok_or(OrderError::Backend("METIS returned a node id out of range"))?;
        perm_out.push(*node);
    }
    Ok(perm_out)
}

/// Run `f`, then put the process's SIGABRT and SIGTERM dispositions back exactly as they
/// were before it — handler, flags and mask.
///
/// Every METIS API call goes through GKlib's `gk_sigtrap` / `gk_siguntrap`
/// (`GKlib/src/error.c`): on entry it installs its own `gk_sigthrow` for SIGABRT and
/// SIGTERM with `signal()`, and on exit it reinstalls the previous handlers with
/// `signal()` again. `signal()` keeps only the handler pointer. It drops the `sigaction`
/// flags and mask the handler was installed with, and installs it with `signal()`'s own
/// flags instead. Redis installs its crash reporter for SIGABRT with `sigaction` and
/// `SA_SIGINFO | SA_NODEFER` (`setupSigSegvHandler`, `src/debug.c`), and its shutdown
/// handler for SIGTERM with no flags (`setupSignalHandlers`, `src/server.c`). After one
/// unguarded METIS call, a later `abort()` anywhere in the process calls the crash
/// reporter as a one-argument handler: it reads the `siginfo_t *` and `ucontext_t *` it
/// was not given, and the crash report is garbage or a second crash. This guard undoes
/// that. (`a_sa_siginfo_handler_survives_the_guarded_call` shows the damage and the fix.)
///
/// What it cannot fix is the window *during* the call, where GKlib's handler is installed
/// for the whole process: a SIGTERM delivered then is swallowed (on the calling thread
/// METIS returns an error and Redis never sees the shutdown request) or undefined (on any
/// other thread `gk_sigthrow` `longjmp`s through a jump buffer that thread never set). The
/// C engine has the same window. Closing it needs a change to GKlib itself.
///
/// It is its own function, called from one place, so it can be removed in one commit
/// the day GKlib stops touching the handlers.
fn preserving_signal_handlers<R>(f: impl FnOnce() -> R) -> R {
    struct Saved([(c_int, libc::sigaction); 2]);
    impl Drop for Saved {
        fn drop(&mut self) {
            for (signal, action) in &self.0 {
                // SAFETY: `action` is a disposition `sigaction` itself reported for
                // `signal`, so it is valid to install again.
                let rc = unsafe { libc::sigaction(*signal, action, null_mut()) };
                debug_assert_eq!(rc, 0, "restoring the handler for signal {signal}");
            }
        }
    }
    let save = |signal: c_int| {
        let mut action = MaybeUninit::<libc::sigaction>::zeroed();
        // SAFETY: a null new action only reads the current one into `action`, which
        // `sigaction` fully writes for a valid signal number.
        let rc = unsafe { libc::sigaction(signal, null_mut(), action.as_mut_ptr()) };
        assert_eq!(rc, 0, "reading the handler for signal {signal}");
        // SAFETY: written by the successful call above (and zeroed before it).
        (signal, unsafe { action.assume_init() })
    };
    let _restore_on_return = Saved([save(libc::SIGABRT), save(libc::SIGTERM)]);
    f()
}

#[cfg(test)]
mod tests {
    use std::ffi::c_int;
    use std::mem::MaybeUninit;
    use std::ptr::null_mut;
    use std::sync::PoisonError;

    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

    use super::{METIS_LOCK, metis_graph, node_nd_unguarded, preserving_signal_handlers};
    use crate::index::falkordb::cch::order::{MetisOrdering, Ordering};
    use crate::index::falkordb::cch::testkit::{grid, random_sparse, to_input};

    extern "C" fn three_argument_handler(
        _: c_int,
        _: *mut libc::siginfo_t,
        _: *mut libc::c_void,
    ) {
    }

    fn current(signal: c_int) -> libc::sigaction {
        let mut action = MaybeUninit::<libc::sigaction>::zeroed();
        // SAFETY: reads the current disposition into `action`.
        assert_eq!(
            unsafe { libc::sigaction(signal, null_mut(), action.as_mut_ptr()) },
            0
        );
        // SAFETY: written by the call above.
        unsafe { action.assume_init() }
    }

    /// What Redis installs for SIGABRT (`setupSigSegvHandler`): a three-argument handler
    /// with `SA_SIGINFO | SA_NODEFER`, here with SIGUSR2 in its mask so the mask is
    /// checked too.
    fn install_redis_like(signal: c_int) {
        let mut action = MaybeUninit::<libc::sigaction>::zeroed();
        // SAFETY: a zeroed `sigaction` filled in field by field, as C callers do.
        unsafe {
            let a = action.as_mut_ptr();
            (*a).sa_sigaction = three_argument_handler as *const () as usize;
            (*a).sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
            libc::sigemptyset(&raw mut (*a).sa_mask);
            libc::sigaddset(&raw mut (*a).sa_mask, libc::SIGUSR2);
            assert_eq!(libc::sigaction(signal, a, null_mut()), 0);
        }
    }

    fn is_redis_like(action: &libc::sigaction) -> bool {
        // SAFETY: `sigismember` only reads the set.
        let masked = unsafe { libc::sigismember(&raw const action.sa_mask, libc::SIGUSR2) };
        action.sa_sigaction == three_argument_handler as *const () as usize
            && action.sa_flags & libc::SA_SIGINFO != 0
            && action.sa_flags & libc::SA_NODEFER != 0
            && masked == 1
    }

    /// The damage, then the fix. A SA_SIGINFO handler for SIGABRT and SIGTERM, then one
    /// METIS call: unguarded, the same handler comes back without SA_SIGINFO (so a later
    /// signal would call it with one argument instead of three) and without its mask;
    /// guarded, it comes back exactly as it was.
    #[test]
    fn a_sa_siginfo_handler_survives_the_guarded_call() {
        let _alone = METIS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let before = [current(libc::SIGABRT), current(libc::SIGTERM)];
        let skeleton = to_input(&grid(8, 8)).skeleton;
        let order = || node_nd_unguarded(metis_graph(&skeleton).expect("valid")).expect("order");

        for signal in [libc::SIGABRT, libc::SIGTERM] {
            install_redis_like(signal);
        }
        let unguarded = order();
        for signal in [libc::SIGABRT, libc::SIGTERM] {
            let after = current(signal);
            assert_eq!(
                after.sa_sigaction, three_argument_handler as *const () as usize,
                "GKlib reinstalls the same function"
            );
            assert_eq!(
                after.sa_flags & libc::SA_SIGINFO,
                0,
                "signal {signal}: unguarded, SA_SIGINFO is expected to be lost"
            );
            assert!(!is_redis_like(&after));
        }

        for signal in [libc::SIGABRT, libc::SIGTERM] {
            install_redis_like(signal);
        }
        let guarded = preserving_signal_handlers(order);
        for signal in [libc::SIGABRT, libc::SIGTERM] {
            assert!(
                is_redis_like(&current(signal)),
                "signal {signal}: the guard must restore handler, flags and mask"
            );
        }
        assert_eq!(unguarded, guarded, "the guard does not change the order");

        for (signal, action) in [libc::SIGABRT, libc::SIGTERM].into_iter().zip(&before) {
            // SAFETY: reinstalls what `sigaction` reported before the test.
            assert_eq!(unsafe { libc::sigaction(signal, action, null_mut()) }, 0);
        }
    }

    /// METIS draws from GKlib's own generator, not the C library's `rand`, which the rest
    /// of the process shares (RediSearch's garbage collector calls it): with another thread
    /// reseeding and drawing from `rand` throughout, every order is the one a quiet process
    /// gives. Built without `GKRAND` this fails within a few calls.
    #[test]
    fn the_order_ignores_other_users_of_libc_rand() {
        for skeleton in [
            to_input(&grid(40, 40)).skeleton,
            to_input(&random_sparse(2_000, 6_000, 9, 11)).skeleton,
        ] {
            let quiet = MetisOrdering.order(&skeleton).expect("order");
            let stop = AtomicBool::new(false);
            let noisy: Vec<Vec<u64>> = std::thread::scope(|s| {
                s.spawn(|| {
                    let mut seed: u32 = 1;
                    while !stop.load(Relaxed) {
                        // SAFETY: `srand` and `rand` take and return plain integers.
                        unsafe {
                            libc::srand(seed);
                            libc::rand();
                        }
                        seed = seed.wrapping_add(1);
                    }
                });
                let orders = (0..20)
                    .map(|_| MetisOrdering.order(&skeleton).expect("order"))
                    .collect();
                stop.store(true, Relaxed);
                orders
            });
            for (k, order) in noisy.iter().enumerate() {
                assert_eq!(*order, quiet, "call {k} drew on the shared generator");
            }
        }
    }
}
