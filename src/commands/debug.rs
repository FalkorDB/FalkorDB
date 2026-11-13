use crate::redis_type::{
    create_virtual_keys, delete_stale_virtual_keys, delete_virtual_keys, finalize_pending_graphs,
};
use crate::serializers::DECODE_STATE;
use redis_module::{
    Context, ContextFlags, NextArg, RedisError, RedisResult, RedisString, RedisValue,
};

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
    // During an async (`swapdb`) replica load Redis serves commands from inside
    // the decode, which holds `DECODE_STATE` across its reads: taking it here
    // would deadlock the server. A replica only gets AUX from its primary's
    // stream, which Redis does not apply until the load is over.
    if ctx.get_flags().contains(ContextFlags::ASYNC_LOADING) {
        return Err(RedisError::Str(
            "GRAPH.DEBUG AUX is not allowed while a replica is loading",
        ));
    }
    let result = match action.to_uppercase().as_str() {
        "START" => {
            DECODE_STATE.lock().clear();
            unsafe { create_virtual_keys(ctx.ctx) };
            Ok(RedisValue::Integer(1))
        }
        "END" => {
            finalize_pending_graphs();
            unsafe {
                delete_stale_virtual_keys(ctx.ctx);
                // Undo START's `create_virtual_keys` the way a finished SAVE
                // does. Left in `VKEY_STATE`, every later save -- a replica's
                // full sync, BGSAVE -- encoded the graph as a multi-key header
                // naming virtual keys that no longer existed, and the loader
                // kept an empty placeholder waiting for them.
                delete_virtual_keys(ctx.ctx);
            }
            Ok(RedisValue::Integer(0))
        }
        _ => Err(RedisError::String(format!("Unknown AUX action: {action}"))),
    };
    ctx.replicate_verbatim();
    result
}
