use crate::{
    commands::EMPTY_KEY_ERR,
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, register_graph, up_to_nul},
    redis_type::GRAPH_TYPE,
    serializers,
};
use graph::graph::graphblas::matrix::set_nthreads;
use graph::graph::mvcc_graph::MvccGraph;
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::os::raw::c_int;
use std::os::unix::io::{FromRawFd, OwnedFd};
use std::sync::Arc;

unsafe extern "C" {
    fn pipe(pipefd: *mut c_int) -> c_int;
    fn fork() -> i32;
    fn _exit(status: c_int) -> !;
    fn waitpid(
        pid: i32,
        status: *mut c_int,
        options: c_int,
    ) -> i32;
}

pub fn graph_copy(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let src_key_name = args.next_arg()?;
    let dest_key_name = args.next_arg()?;

    let dest_name = std::str::from_utf8(dest_key_name.as_slice())
        .map_err(|_| RedisError::Str("ERR destination key is not valid UTF-8"))?;

    // The destination *key* is the full bytes the command named, for the existence
    // check as much as for the write: C opens `copy_ctx->rm_dest` both times and never
    // rebuilds it from the name, so `c_graph_key` has no business here. Guarding one
    // key and writing another would let `GRAPH.COPY src "victim\0bypass"` pass a check
    // on the empty `victim\0bypass` and then overwrite the live `victim`.
    //
    // The graph's own *name* is truncated, which is what C does in the forked child:
    // `GraphContext_Rename(ctx, gc, copy_ctx->dest)` renames it to a C string, so the
    // truncated name is what lands in the dump header the decoder reads back.
    let graph_name = up_to_nul(dest_name);

    // Open src key (read) and verify it holds a graph.
    let src_key = ctx.open_key(&src_key_name);
    let src_graph = match src_key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        Some(g) => g.clone(),
        None => return EMPTY_KEY_ERR,
    };

    // Verify dest key does not already exist.
    let dest_key = ctx.open_key_writable(&dest_key_name);
    if dest_key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .is_some()
    {
        return Err(RedisError::Str("destination key already exists"));
    }
    // Also check if dest key exists as any other type.
    if dest_key.key_type() != redis_module::KeyType::Empty {
        return Err(RedisError::Str("destination key already exists"));
    }

    let cache_size = *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize;

    // Read-lock the source graph for the duration of the copy.
    let tg = src_graph.read();
    let g = tg.graph.read();
    let graph = g.borrow();

    // Create pipe.
    let mut pipe_fds = [0i32; 2];
    if unsafe { pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return Err(RedisError::Str("ERR could not create pipe"));
    }
    // SAFETY: pipe() just returned two valid, open file descriptors.
    let read_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[0]) };
    let write_fd = unsafe { OwnedFd::from_raw_fd(pipe_fds[1]) };

    // Fork.
    let pid = unsafe { fork() };
    if pid < 0 {
        // OwnedFd drop closes both fds automatically.
        drop(read_fd);
        drop(write_fd);
        return Err(RedisError::Str("ERR could not fork"));
    }

    if pid == 0 {
        // --- Child process ---
        drop(read_fd);
        set_nthreads(1);

        serializers::encoder::pipe_save_graph(write_fd, &graph);

        // _exit to avoid running destructors in the child.
        unsafe { _exit(0) };
    }

    // --- Parent process ---
    drop(write_fd);

    // Drop locks before blocking on pipe read (child has its own CoW copy).
    drop(graph);
    drop(g);
    drop(tg);

    let result = serializers::decoder::pipe_load_graph(read_fd, cache_size, graph_name);

    // Wait for child, retrying on EINTR.
    let mut status = 0i32;
    loop {
        let ret = unsafe { waitpid(pid, &raw mut status, 0) };
        if ret == pid {
            break;
        }
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(RedisError::String(format!("ERR waitpid failed: {err}")));
        }
    }

    let new_graph = result.map_err(RedisError::String)?;

    // Check child exit status.
    // WIFEXITED: (status & 0x7f) == 0
    // WEXITSTATUS: (status >> 8) & 0xff
    let exited = status.trailing_zeros() >= 7;
    let exit_code = (status >> 8) & 0xff;
    if !exited || exit_code != 0 {
        return Err(RedisError::Str("ERR child process failed during copy"));
    }

    // Serialize the decoded graph for replication before wrapping.
    let serialized = serializers::encoder::vec_save_graph(&new_graph);

    // Wrap the decoded graph and set on dest key.
    let mvcc = MvccGraph::from_graph(new_graph);
    let graph_arc = mvcc.read();
    graph_arc.borrow().set_indexer_graph(graph_arc.clone());
    let tg = ThreadedGraph::from_mvcc(mvcc);
    let boxed = Arc::new(RwLock::new(tg));

    dest_key.set_value(&GRAPH_TYPE, boxed.clone())?;
    // Registered under the key it was written to: the registry is the module's index of
    // the *keyspace*, and `graph_rdb_save`, the virtual-key builder and the telemetry
    // flusher all address Redis by the name they find here.
    register_graph(dest_name.to_string(), boxed);

    // Replicate via GRAPH.RESTORE so replicas receive the serialized graph data. The
    // full key bytes, not `graph_name`: C sends the truncated C string here and leaves
    // the replica holding a different key than the master, which is a divergence rather
    // than a behaviour worth reproducing.
    ctx.replicate("GRAPH.RESTORE", &[dest_name.as_bytes(), &serialized]);

    Ok(RedisValue::SimpleStringStatic("OK"))
}
