use crate::{config::CONFIGURATION_CACHE_SIZE, graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
use graph::entity_type::EntityType;
use graph::graph::constraint::ConstraintType;
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

/// Validate that a name is a valid identifier: starts with a letter or underscore,
/// followed by letters, digits, or underscores.
fn is_valid_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn graph_constraint(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    // GRAPH.CONSTRAINT CREATE|DROP <key> UNIQUE|MANDATORY NODE|RELATIONSHIP <label> PROPERTIES <count> <prop1>...
    let mut args = args.into_iter().skip(1);

    // Operation: CREATE or DROP
    let op_str = args.next_str()?;
    let is_create = match op_str.to_uppercase().as_str() {
        "CREATE" => true,
        "DROP" => false,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint operation".into(),
            ));
        }
    };

    // Graph key
    let key_str = args.next_arg()?;

    // Constraint type
    let ct_str = args.next_str()?;
    let ct = match ct_str.to_uppercase().as_str() {
        "UNIQUE" => ConstraintType::Unique,
        "MANDATORY" => ConstraintType::Mandatory,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint type".into(),
            ));
        }
    };

    // Entity type
    let et_str = args.next_str()?;
    let entity_type = match et_str.to_uppercase().as_str() {
        "NODE" | "LABEL" => EntityType::Node,
        "RELATIONSHIP" | "EDGE" => EntityType::Relationship,
        _ => {
            return Err(redis_module::RedisError::String(
                "Invalid constraint entity type".into(),
            ));
        }
    };

    // Label name
    let label_str = args.next_str()?;
    if !is_valid_identifier(label_str) {
        return Err(redis_module::RedisError::String(format!(
            "Label name {label_str} is invalid"
        )));
    }
    let label = Arc::new(label_str.to_string());

    // PROPERTIES keyword
    let props_kw = args.next_str()?;
    if props_kw.to_uppercase() != "PROPERTIES" {
        return Err(redis_module::RedisError::String(
            "Expected PROPERTIES keyword".into(),
        ));
    }

    // Property count
    let prop_count_str = args.next_str()?;
    let prop_count: i64 = prop_count_str.parse().map_err(|_| {
        redis_module::RedisError::String(
            "Number of properties must be an integer between 1 and 255".into(),
        )
    })?;
    if !(1..=255).contains(&prop_count) {
        return Err(redis_module::RedisError::String(
            "Number of properties must be an integer between 1 and 255".into(),
        ));
    }

    // Property names
    let mut properties = Vec::with_capacity(prop_count as usize);
    for _ in 0..prop_count {
        let prop_str = args.next_str()?;
        if !is_valid_identifier(prop_str) {
            return Err(redis_module::RedisError::String(format!(
                "Property name {prop_str} is invalid"
            )));
        }
        let prop = Arc::new(prop_str.to_string());
        if properties.contains(&prop) {
            return Err(redis_module::RedisError::String(
                "Properties cannot contain duplicates".into(),
            ));
        }
        properties.push(prop);
    }

    // Reject trailing tokens
    if args.next().is_some() {
        return Err(redis_module::RedisError::String(
            "Unexpected extra arguments".into(),
        ));
    }

    // Open or create graph
    let key = ctx.open_key_writable(&key_str);
    let graph = if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        if !is_create {
            return Err(redis_module::RedisError::String(
                "Unable to drop constraint, no such constraint.".into(),
            ));
        }
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        key.set_value(&GRAPH_TYPE, g.clone())?;
        g
    };

    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR write lock unavailable".into(),
        ));
    };

    let is_replicated = ctx.get_flags().contains(ContextFlags::REPLICATED);

    let result: Result<bool, String> = {
        let mut g = g_arc.borrow_mut();
        if is_create {
            g.create_constraint(ct, entity_type, label, properties)
        } else {
            g.drop_constraint(&ct, &entity_type, &label, &properties)
                .map(|()| false)
        }
    };

    match result {
        Ok(needs_async_validation) => {
            tg.graph.commit(g_arc);

            // Spawn background validation for large datasets
            if needs_async_validation {
                let graph_clone = graph.clone();
                graph::threadpool::spawn(
                    move || {
                        let mut tg = graph_clone.write();
                        if let Some(g_arc) = tg.graph.write() {
                            g_arc.borrow_mut().validate_pending_constraints();
                            tg.graph.commit(g_arc);
                        }
                    },
                    Some(0),
                );
            }

            if !is_replicated {
                ctx.replicate_verbatim();
                if is_create {
                    // Replicate a second time to signal constraint activation,
                    // matching C FalkorDB's two-phase create + activate protocol.
                    ctx.replicate_verbatim();
                }
            }
            let value = tg.graph.read().borrow().maybe_flush_caches();
            if let Err(e) = value {
                ctx.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
            }
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) if is_replicated && e == "Constraint already exists" => {
            // Activation signal on replica — constraint already created by
            // the first replicated command, silently succeed.
            tg.graph.rollback();
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) => {
            tg.graph.rollback();
            Err(redis_module::RedisError::String(e))
        }
    }
}
