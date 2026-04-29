use crate::{
    commands::EMPTY_KEY_ERR, config::CONFIGURATION_CACHE_SIZE, graph_core::ThreadedGraph,
    redis_type::GRAPH_TYPE, serializers,
};
use graph::graph::graphblas::matrix::set_nthreads;
use graph::graph::mvcc_graph::MvccGraph;
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::os::raw::c_int;
use std::sync::Arc;

unsafe extern "C" {
    fn pipe(pipefd: *mut c_int) -> c_int;
    fn fork() -> i32;
    fn close(fd: c_int) -> c_int;
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

    let dest_name = dest_key_name.to_string_lossy();

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
        return Err(RedisError::Str("ERR destination key already exists"));
    }
    // Also check if dest key exists as any other type.
    if dest_key.key_type() != redis_module::KeyType::Empty {
        return Err(RedisError::Str("ERR destination key already exists"));
    }

    let cache_size = *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize;

    // Read-lock the source graph for the duration of the copy.
    let tg = src_graph.read();
    let g = tg.graph.read();
    let graph = g.borrow();

    // Build attribute snapshots before fork (fork-safe attribute reading).
    let snapshots = if graph.needs_rdb_snapshot() {
        Some(Arc::new(graph.build_rdb_snapshots()))
    } else {
        None
    };

    // Create pipe.
    let mut pipe_fds = [0i32; 2];
    if unsafe { pipe(pipe_fds.as_mut_ptr()) } != 0 {
        return Err(RedisError::Str("ERR could not create pipe"));
    }
    let read_fd = pipe_fds[0];
    let write_fd = pipe_fds[1];

    // Fork.
    let pid = unsafe { fork() };
    if pid < 0 {
        unsafe {
            close(read_fd);
            close(write_fd);
        }
        return Err(RedisError::Str("ERR could not fork"));
    }

    if pid == 0 {
        // --- Child process ---
        unsafe { close(read_fd) };
        set_nthreads(1);

        serializers::encoder::pipe_save_graph(
            write_fd,
            &graph,
            snapshots.as_ref().map(AsRef::as_ref),
        );

        // _exit to avoid running destructors in the child.
        unsafe { _exit(0) };
    }

    // --- Parent process ---
    unsafe { close(write_fd) };

    // Drop locks before blocking on pipe read (child has its own CoW copy).
    drop(graph);
    drop(g);
    drop(tg);

    let result = serializers::decoder::pipe_load_graph(read_fd, cache_size, &dest_name);

    // Wait for child.
    let mut status = 0i32;
    unsafe { waitpid(pid, &raw mut status, 0) };

    let new_graph = result.map_err(RedisError::String)?;

    // Check child exit status.
    // WIFEXITED: (status & 0x7f) == 0
    // WEXITSTATUS: (status >> 8) & 0xff
    let exited = (status & 0x7f) == 0;
    let exit_code = (status >> 8) & 0xff;
    if !exited || exit_code != 0 {
        return Err(RedisError::Str("ERR child process failed during copy"));
    }

    // Serialize the decoded graph for replication before wrapping.
    let serialized = serializers::encoder::vec_save_graph(&new_graph, None);

    // Wrap the decoded graph and set on dest key.
    let mvcc = MvccGraph::from_graph(new_graph);
    let graph_arc = mvcc.read();
    graph_arc.borrow_mut().set_indexer_graph(graph_arc.clone());
    let tg = ThreadedGraph::from_mvcc(mvcc);
    let boxed = Arc::new(RwLock::new(tg));

    dest_key.set_value(&GRAPH_TYPE, boxed)?;

    // Replicate via GRAPH.RESTORE so replicas receive the serialized graph data.
    ctx.replicate("GRAPH.RESTORE", &[dest_name.as_bytes(), &serialized]);

    Ok(RedisValue::SimpleStringStatic("OK"))
}
