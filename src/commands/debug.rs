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
    match action.to_uppercase().as_str() {
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
    }
}
