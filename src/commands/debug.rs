use crate::redis_type::{create_virtual_keys, delete_stale_virtual_keys, finalize_pending_graphs};
use crate::serializers::DECODE_STATE;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};

pub fn graph_debug(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }
    let mut args_iter = args.into_iter().skip(1);
    let subcmd = args_iter.next_str()?;

    match subcmd.to_uppercase().as_str() {
        "AUX" => debug_aux(ctx, args_iter),
        #[cfg(feature = "index-population-hooks")]
        "INDEX_POPULATION" => debug_index_population(args_iter),
        _ => Err(RedisError::String(format!(
            "Unknown DEBUG subcommand: {subcmd}"
        ))),
    }
}

fn debug_aux(
    ctx: &Context,
    mut args: impl Iterator<Item = RedisString>,
) -> RedisResult {
    let action = args.next_str()?;
    let result = match action.to_uppercase().as_str() {
        "START" => {
            DECODE_STATE.lock().clear();
            unsafe { create_virtual_keys(ctx.ctx) };
            Ok(RedisValue::Integer(1))
        }
        "END" => {
            finalize_pending_graphs();
            unsafe { delete_stale_virtual_keys(ctx.ctx) };
            Ok(RedisValue::Integer(0))
        }
        _ => Err(RedisError::String(format!("Unknown AUX action: {action}"))),
    };
    ctx.replicate_verbatim();
    result
}

/// `GRAPH.DEBUG INDEX_POPULATION <action> [n]`, only in builds with the
/// `index-population-hooks` feature (see `graph::index::population_hooks`):
///
/// * `BATCH_SIZE n`: entities per population batch (0 restores the default);
/// * `PAUSE` / `RESUME`: gate population; while paused a batch waits for a token;
/// * `STEP [n]`: let `n` (default 1) batches through;
/// * `BATCHES`: batches completed so far;
/// * `WINDOWS`: `[fired, timed_out]` windows (see `WINDOW`);
/// * `WINDOW n`: arm the next writer: after it publishes its index documents and
///   before it commits, `n` population batches run (and are waited for, up to
///   2 s); `WINDOW 0` disarms.
#[cfg(feature = "index-population-hooks")]
fn debug_index_population(mut args: impl Iterator<Item = RedisString>) -> RedisResult {
    use graph::index::population_hooks as hooks;

    let action = args.next_str()?.to_uppercase();
    match action.as_str() {
        "BATCH_SIZE" => {
            hooks::set_batch_size(args.next_u64()? as usize);
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        "PAUSE" => {
            hooks::set_paused(true);
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        "RESUME" => {
            hooks::set_paused(false);
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        "STEP" => {
            // `STEP` alone is one batch; a bad count is an error, not a default.
            let n = match args.next() {
                None => 1,
                Some(n) => n.parse_unsigned_integer()?,
            };
            hooks::step(n);
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        "BATCHES" => Ok(RedisValue::Integer(hooks::batches_done() as i64)),
        "WINDOWS" => {
            let (fired, timed_out) = hooks::windows();
            Ok(RedisValue::Array(vec![
                RedisValue::Integer(fired as i64),
                RedisValue::Integer(timed_out as i64),
            ]))
        }
        "WINDOW" => {
            hooks::set_window_batches(args.next_u64()?);
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        _ => Err(RedisError::String(format!(
            "Unknown INDEX_POPULATION action: {action}"
        ))),
    }
}
