//! What to do when a replicated command will not apply.
//!
//! A command reaches the replication stream — or the AOF — only after it
//! succeeded on the master. So a failure to apply it here does not mean the
//! command was bad; it means this instance's dataset no longer matches the
//! state that command was recorded against. The write is already lost, and
//! every later effect is being applied to a graph that has silently drifted.
//!
//! Redis does not break a replication link over a module command's error
//! reply: the offset advances and the stream keeps flowing. So detecting
//! divergence and returning an error, which is all this module used to do,
//! leaves the replica serving wrong data indefinitely. Effects v3 makes that
//! worse rather than better — it adds thirteen distinct divergence checks
//! where v2 had almost none, so the case is far more reachable.
//!
//! Mirrors C's `DivergenceGuard_OnFailure` (`src/replication/divergence_guard.c`
//! on `master`), including its three failure modes, because a replica has to
//! behave the same way whichever engine is running it.

// `log_warning` is the most severe level Redis offers a module: `RedisLogLevel`
// is Debug | Notice | Verbose | Warning, and the binding maps `log::Level::Error`
// onto Warning. Redis prints it with `#`, which is what an operator greps for.
use redis_module::logging::log_warning;
use redis_module::{Context, ContextFlags};
use std::time::Duration;

/// Whether a failure of this command means divergence rather than a bad
/// request.
///
/// Only a command the master already accepted can prove this instance wrong.
/// A `GRAPH.EFFECT` sent straight from a client is just a malformed command —
/// answering it with a forced resync, or with `exit(1)`, would hand any client
/// a way to take a replica down, and this is a payload any client can send.
///
/// C guards `GRAPH.EFFECT` unconditionally
/// (`cmd_effect.c`); this is a deliberate difference, for that reason.
#[must_use]
pub fn is_replayed(ctx: &Context) -> bool {
    ctx.get_flags()
        .intersects(ContextFlags::REPLICATED | ContextFlags::LOADING)
}

/// Called when a command replicated from the master, or replayed from local
/// AOF/RDB, fails to apply.
///
/// Never returns normally in the loading case: there is nothing a resync can
/// repair when the divergence is already baked into this instance's own
/// persisted state, and the rest of the file would keep replaying against an
/// already-wrong dataset while a fix was pending.
pub fn on_failure(
    ctx: &Context,
    graph_name: &str,
    cmd_name: &str,
    detail: &str,
) {
    // The gate lives here rather than at the call site. A caller that forgets it
    // hands any client a way to force a replica to resync — or, under `LOADING`,
    // to `exit(1)` — by sending one malformed `GRAPH.EFFECT`. That is not a
    // check to leave as a convention.
    if !is_replayed(ctx) {
        return;
    }
    if ctx.get_flags().contains(ContextFlags::LOADING) {
        log_warning(format!(
            "Diverged applying {cmd_name} on graph '{graph_name}' while loading from disk: \
             {detail}. A full resync cannot repair already-loaded state, shutting down."
        ));
        std::process::exit(1);
    }

    log_warning(format!(
        "Replica diverged from master applying {cmd_name} on graph '{graph_name}': {detail}. \
         Scheduling a forced full resync with master."
    ));

    // Deferred rather than done here: this runs inside the command handler,
    // holding the graph's locks, and `REPLICAOF` tears down the replication
    // link the caller is being served from.
    let graph_name = graph_name.to_string();
    ctx.create_timer(
        Duration::from_millis(0),
        |ctx: &Context, graph_name: String| force_full_resync(ctx, &graph_name),
        graph_name,
    );
}

/// `REPLICAOF NO ONE` then `REPLICAOF <master>`, which is what makes the
/// reconnect a `FULLRESYNC`.
///
/// The first call is the load-bearing one: it discards this replica's cached
/// replication ID and offset, so the master cannot satisfy the reconnect with a
/// partial resync (`PSYNC CONTINUE`) against the dataset that just diverged.
///
/// Every failure path exits. A replica that cannot re-sync must not keep
/// serving data it knows is wrong.
fn force_full_resync(
    ctx: &Context,
    graph_name: &str,
) {
    let Some((host, port)) = master_address(ctx) else {
        log_warning(format!(
            "Unable to determine master address for graph '{graph_name}', shutting down \
             instead of forcing a full resync"
        ));
        std::process::exit(1);
    };

    if let Err(e) = ctx.call("REPLICAOF", &["NO", "ONE"]) {
        log_warning(format!(
            "REPLICAOF NO ONE failed for graph '{graph_name}' ({e}), shutting down instead \
             of forcing a full resync"
        ));
        std::process::exit(1);
    }

    if let Err(e) = ctx.call("REPLICAOF", &[host.as_str(), port.as_str()]) {
        log_warning(format!(
            "Failed to reattach to master {host}:{port} for graph '{graph_name}' ({e}), \
             shutting down"
        ));
        std::process::exit(1);
    }

    redis_module::logging::log_notice(format!(
        "Forced full resync with master {host}:{port} initiated after divergence detected \
         on graph '{graph_name}'"
    ));
}

/// The master this replica is attached to, from `INFO replication`.
///
/// `None` when either field is missing, which is what a master — or an
/// instance mid-topology-change — reports.
fn master_address(ctx: &Context) -> Option<(String, String)> {
    let info = ctx.server_info("Replication");
    let host = info.field("master_host")?.to_string();
    let port = info.field("master_port")?.to_string();
    if host.is_empty() || port.is_empty() {
        return None;
    }
    Some((host, port))
}
