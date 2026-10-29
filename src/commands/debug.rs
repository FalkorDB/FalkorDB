use crate::redis_type::{create_virtual_keys, delete_stale_virtual_keys, finalize_pending_graphs};
use crate::serializers::DECODE_STATE;
use valkey_module::{Context, NextArg, ValkeyError, ValkeyResult, ValkeyString, ValkeyValue};

pub fn graph_debug(
    ctx: &Context,
    args: Vec<ValkeyString>,
) -> ValkeyResult {
    if args.len() < 3 {
        return Err(ValkeyError::WrongArity);
    }
    let mut args_iter = args.into_iter().skip(1);
    let subcmd = args_iter.next_str()?;

    match subcmd.to_uppercase().as_str() {
        "AUX" => debug_aux(ctx, args_iter),
        _ => Err(ValkeyError::String(format!(
            "Unknown DEBUG subcommand: {subcmd}"
        ))),
    }
}

fn debug_aux(
    ctx: &Context,
    mut args: impl Iterator<Item = ValkeyString>,
) -> ValkeyResult {
    let action = args.next_str()?;
    let result = match action.to_uppercase().as_str() {
        "START" => {
            DECODE_STATE.lock().clear();
            unsafe { create_virtual_keys(ctx.ctx) };
            Ok(ValkeyValue::Integer(1))
        }
        "END" => {
            finalize_pending_graphs();
            unsafe { delete_stale_virtual_keys(ctx.ctx) };
            Ok(ValkeyValue::Integer(0))
        }
        _ => Err(ValkeyError::String(format!("Unknown AUX action: {action}"))),
    };
    ctx.replicate_verbatim();
    result
}
