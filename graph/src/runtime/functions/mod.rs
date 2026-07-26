//! Built-in Cypher function registry and type definitions.
//!
//! This module is the root of the functions subsystem. It owns the global
//! function registry and the core types that every other module in the
//! crate depends on (`FnType`, `Type`, `GraphFn`, ...).  The actual
//! function implementations live in categorised submodules:
//!
//! ```text
//! functions/
//! +----- mod.rs          Registry, types, init_functions()
//! |
//! +----- entity.rs       Graph-entity inspection (labels, id, properties, ...)
//! +----- string.rs       String manipulation (toLower, substring, replace, ...)
//! +----- math.rs         Numeric / general math (abs, ceil, sqrt, range, ...)
//! +----- trig.rs         Trigonometric functions (sin, cos, atan2, ...)
//! +----- conversion.rs   Type casting (toInteger, toFloat, toString, ...)
//! +----- list.rs         List operations (head, tail, size, reverse, ...)
//! +----- aggregation.rs  Aggregation functions (count, sum, avg, collect, ...)
//! +----- spatial.rs      Spatial / vector (point, distance, vecf32)
//! +----- path.rs         Path decomposition (nodes, relationships)
//! +----- internal.rs     Internal-only operators (starts_with, case, ...)
//! +----- procedures.rs   CALL procedures (db.labels, db.indexes, ...)
//! ```
//!
//! ## Function registry
//!
//! [`init_functions`] populates a `OnceLock<Functions>` singleton at startup.
//! Each entry maps a **lowercase** function name to an `Arc<GraphFn>` that
//! stores the implementation pointer, argument types, return type, and whether
//! the function is scalar / aggregation / procedure / internal.
//!
//! ```text
//! +--------------------------+
//! |  OnceLock<Functions>     |
//! |  HashMap<String,         |
//! |          Arc<GraphFn>>   |
//! +------+-------------------+
//!        |
//!        |  "tolower" --> GraphFn { func: string::string_to_lower, ... }
//!        |  "count"   --> GraphFn { func: aggregation::count,      ... }
//!        |  "db.labels" -> GraphFn { func: procedures::db_labels,  ... }
//!        |  ...
//! ```
//!
//! Callers obtain a reference via [`get_functions`] and look up entries
//! with `Functions::get(name, &FnType)`.
//!
//! ## Key types
//!
//! | Type          | Purpose                                              |
//! |---------------|------------------------------------------------------|
//! | [`FnType`]    | Discriminates Function / Internal / Procedure / Agg  |
//! | [`Type`]      | Cypher type system (Int, String, Union, Optional ...) |
//! | [`GraphFn`]   | A registered function: name + pointer + metadata     |
//! | [`Functions`] | The registry itself (wraps a `HashMap`)               |

/// Defines a Cypher runtime function and registers it in a single declaration.
///
/// Invoke inside a `register(funcs: &mut Functions)` function.  The macro
/// defines a nested function with the `RuntimeFn` signature and emits the
/// matching `funcs.add()` or `funcs.add_var_len()` call.
///
/// # Arms
///
/// | Discriminator           | Pattern               | `FnType` produced       |
/// |-------------------------|-----------------------|-------------------------|
/// | *(none)*                | scalar, fixed args    | `Function`              |
/// | `var_arg:`              | scalar, var-len args  | `Function`              |
/// | `agg_init:`             | aggregation, no final | `Aggregation(v, None)`  |
/// | `agg_init:` `finalizer:`| aggregation + final   | `Aggregation(v, Some)`  |
/// | `internal,`             | internal operator     | `Internal`              |
/// | `procedure:`            | read-only procedure   | `Procedure(yields)`     |
/// | `write procedure:`      | write procedure       | `Procedure(yields)`     |
macro_rules! cypher_fn {
    // ── Non-deterministic scalar function (fixed args) ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     non_deterministic,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add(
            $name,
            $fn_name,
            false,
            true,
            vec![$($arg),*],
            FnType::Function,
            $ret,
        );
    };

    // ── Non-deterministic variable-length argument function ──
    ($funcs:ident, $name:expr,
     var_arg: $arg_type:expr,
     ret: $ret:expr,
     non_deterministic,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add_var_len(
            $name,
            $fn_name,
            false,
            true,
            $arg_type,
            FnType::Function,
            $ret,
        );
    };

    // ── Scalar function (FnType::Function, write=false, fixed args) ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add(
            $name,
            $fn_name,
            false,
            false,
            vec![$($arg),*],
            FnType::Function,
            $ret,
        );
    };

    // ── Variable-length argument function (add_var_len) ──
    ($funcs:ident, $name:expr,
     var_arg: $arg_type:expr,
     ret: $ret:expr,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add_var_len(
            $name,
            $fn_name,
            false,
            false,
            $arg_type,
            FnType::Function,
            $ret,
        );
    };

    // ── Aggregation without finalizer ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     agg_init: $init:expr,
     $(batch_agg: $batch:expr,)?
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        #[allow(unused_mut, unused_assignments)]
        let mut __batch: Option<crate::runtime::functions::BatchAggFn> = None;
        $( __batch = Some($batch); )?

        $funcs.add(
            $name,
            $fn_name,
            false,
            false,
            vec![$($arg),*],
            FnType::Aggregation { initial: $init, finalizer: None, batch_agg: __batch },
            $ret,
        );
    };

    // ── Aggregation with finalizer ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     agg_init: $init:expr,
     finalizer: $finalizer:expr,
     $(batch_agg: $batch:expr,)?
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        #[allow(unused_mut, unused_assignments)]
        let mut __batch: Option<crate::runtime::functions::BatchAggFn> = None;
        $( __batch = Some($batch); )?

        $funcs.add(
            $name,
            $fn_name,
            false,
            false,
            vec![$($arg),*],
            FnType::Aggregation { initial: $init, finalizer: Some(Box::new($finalizer)), batch_agg: __batch },
            $ret,
        );
    };

    // ── Internal variable-length argument function (FnType::Internal) ──
    ($funcs:ident, $name:expr,
     var_arg: $arg_type:expr,
     ret: $ret:expr,
     internal,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add_var_len(
            $name,
            $fn_name,
            false,
            false,
            $arg_type,
            FnType::Internal,
            $ret,
        );
    };

    // ── Internal function (FnType::Internal) ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     internal,
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        #[allow(clippy::missing_const_for_fn)]
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<Value, String>
        $body

        $funcs.add(
            $name,
            $fn_name,
            false,
            false,
            vec![$($arg),*],
            FnType::Internal,
            $ret,
        );
    };

    // ── Read-only procedure ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     procedure: [$($yield_col:expr),* $(,)?],
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<crate::runtime::functions::ProcedureBatch, String>
        $body

        $funcs.add_procedure(
            $name,
            $fn_name,
            false,
            false,
            vec![$($arg),*],
            FnType::Procedure(vec![$(String::from($yield_col)),*]),
            $ret,
        );
    };

    // ── Write procedure ──
    ($funcs:ident, $name:expr,
     args: [$($arg:expr),* $(,)?],
     ret: $ret:expr,
     write procedure: [$($yield_col:expr),* $(,)?],
     $(#[$attr:meta])*
     fn $fn_name:ident($rt:pat, $args:pat) $body:block
    ) => {
        $(#[$attr])*
        fn $fn_name(
            $rt: &Runtime,
            $args: &[Value],
        ) -> Result<crate::runtime::functions::ProcedureBatch, String>
        $body

        $funcs.add_procedure(
            $name,
            $fn_name,
            true,
            false,
            vec![$($arg),*],
            FnType::Procedure(vec![$(String::from($yield_col)),*]),
            $ret,
        );
    };
}

mod aggregation;
mod algo_procedures;
mod conversion;
mod entity;
mod internal;
mod list;
mod math;
mod path;
mod procedures;
pub mod spatial;
mod string;
pub mod temporal;
mod trig;

pub use math::apply_pow;
pub(crate) use string::regex_captures_list;

use crate::runtime::{
    batch::{Batch, BatchBuilder},
    runtime::Runtime,
    value::{Value, ValueTypeOf},
};
use parking_lot::RwLock;
use std::{
    borrow::Borrow,
    collections::HashMap,
    fmt::{Debug, Display},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use thin_vec::ThinVec;

pub type ProcedureBatch = Batch<'static>;

#[must_use]
pub fn empty_procedure_batch() -> ProcedureBatch {
    BatchBuilder::new().finish()
}

/// Function type for runtime function implementations.
///
/// Two variants because UDFs need a captured name (closure-shaped) while
/// native functions are plain `fn` pointers. An enum avoids the
/// `Arc<dyn Fn>` indirection and vtable dispatch on the hot path.
pub enum RuntimeFn {
    Native(fn(&Runtime, &[Value]) -> Result<Value, String>),
    NativeProcedureBatch(fn(&Runtime, &[Value]) -> Result<ProcedureBatch, String>),
    Udf(String),
}

impl RuntimeFn {
    #[inline]
    pub fn call(
        &self,
        rt: &Runtime,
        args: &[Value],
    ) -> Result<Value, String> {
        match self {
            Self::Native(f) => f(rt, args),
            Self::NativeProcedureBatch(_) => {
                Err("Procedure runtime function cannot be called as scalar".into())
            }
            Self::Udf(name) => crate::udf::js_context::call_udf_bridge(name, rt, args),
        }
    }

    #[inline]
    pub fn call_procedure_batch(
        &self,
        rt: &Runtime,
        args: &[Value],
    ) -> Result<ProcedureBatch, String> {
        match self {
            Self::NativeProcedureBatch(f) => f(rt, args),
            _ => Err("Function is not a procedure runtime function".into()),
        }
    }
}

/// Optional bulk-aggregation entry point.
///
/// Receives the column of input
/// values for one batch (`inputs`), the number of rows being aggregated
/// (`num_rows` — used by no-input aggregates like `count(*)` where
/// `inputs` is empty), and the previous accumulator. Returns the new
/// accumulator.
///
/// When supplied, the keyless-single-aggregate path in
/// `AggregateOp` skips per-row evaluation and calls this once per batch.
pub type BatchAggFn =
    fn(&Runtime, inputs: &[Value], num_rows: usize, acc: Value) -> Result<Value, String>;

/// Classification of function types.
pub enum FnType {
    /// Regular scalar function (e.g., `toUpper()`)
    Function,
    /// Internal function not exposed to users
    Internal,
    /// Procedure that returns a result set (e.g., `db.labels()`)
    Procedure(Vec<String>),
    /// Aggregation function with initial value and optional finalizer
    Aggregation {
        initial: Value,
        finalizer: Option<Box<dyn Fn(Value) -> Value + Send + Sync>>,
        /// Optional bulk path used by the vectorized aggregate operator
        /// for keyless single-aggregate queries. When `None`, the
        /// per-row `func` is called instead.
        batch_agg: Option<BatchAggFn>,
    },
    /// User-defined function routed through the UDF bridge
    Udf,
}

#[cfg_attr(tarpaulin, skip)]
impl Debug for FnType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Function => write!(f, "Function"),
            Self::Internal => write!(f, "Internal"),
            Self::Procedure(_) => write!(f, "Procedure"),
            Self::Aggregation { .. } => write!(f, "Aggregation"),
            Self::Udf => write!(f, "Udf"),
        }
    }
}

impl PartialEq for FnType {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        matches!(
            (self, other),
            (Self::Function, Self::Function)
                | (Self::Internal, Self::Internal)
                | (Self::Procedure(_), Self::Procedure(_))
                | (Self::Aggregation { .. }, Self::Aggregation { .. })
                | (Self::Udf, Self::Udf)
        )
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Type {
    Null,
    Bool,
    Int,
    Float,
    String,
    List(Box<Self>),
    Map,
    Node,
    Relationship,
    Path,
    VecF32,
    Point,
    Datetime,
    Date,
    Time,
    Duration,
    Any,
    /// `ThinVec` (single pointer, len/cap in the heap header) keeps this
    /// variant pointer-sized so `Type` is 16 bytes instead of 32; `Type`
    /// is embedded in every `Variable`, which in turn caps
    /// `ExprIR<Variable>` — the node type of every cached-plan expression
    /// tree. Construct via [`Type::union`].
    Union(ThinVec<Self>),
    Optional(Box<Self>),
}

impl Type {
    #[must_use]
    pub fn union(types: impl IntoIterator<Item = Self>) -> Self {
        Self::Union(types.into_iter().collect())
    }

    /// Returns true if this type can include a boolean value.
    /// This mirrors `AR_EXP_ReturnsBoolean` in the C implementation:
    /// returns true if the type is Bool, Null, Any, or a Union/Optional
    /// containing Bool/Null/Any.
    #[must_use]
    pub fn can_return_boolean(&self) -> bool {
        match self {
            Self::Bool | Self::Null | Self::Any => true,
            Self::Union(types) => types.iter().any(Self::can_return_boolean),
            Self::Optional(inner) => inner.can_return_boolean(),
            _ => false,
        }
    }

    #[must_use]
    pub fn can_return_entity(&self) -> bool {
        match self {
            Self::Node | Self::Relationship | Self::Path | Self::Any => true,
            Self::Union(types) => types.iter().any(Self::can_return_entity),
            Self::Optional(inner) => inner.can_return_entity(),
            _ => false,
        }
    }

    /// Check if a statically known type is compatible with an expected type.
    #[must_use]
    pub fn is_compatible_with(
        &self,
        expected: &Self,
    ) -> bool {
        match (self, expected) {
            // Unknown types are always compatible at compile time.
            // Null is always compatible (functions accept nullable args).
            (Self::Any | Self::Null, _) | (_, Self::Any) => true,
            // Same concrete type.
            _ if self == expected => true,
            // Union: compatible if any variant matches.
            (_, Self::Union(types)) => types.iter().any(|t| self.is_compatible_with(t)),
            // Optional: compatible with inner type.
            (_, Self::Optional(inner)) => self.is_compatible_with(inner),
            _ => false,
        }
    }
}

#[cfg_attr(tarpaulin, skip)]
impl Display for Type {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "Null"),
            Self::Bool => write!(f, "Boolean"),
            Self::Int => write!(f, "Integer"),
            Self::Float => write!(f, "Float"),
            Self::String => write!(f, "String"),
            Self::List(_) => write!(f, "List"),
            Self::Map => write!(f, "Map"),
            Self::Node => write!(f, "Node"),
            Self::Relationship => write!(f, "Edge"),
            Self::Path => write!(f, "Path"),
            Self::VecF32 => write!(f, "VecF32"),
            Self::Point => write!(f, "Point"),
            Self::Datetime => write!(f, "Datetime"),
            Self::Date => write!(f, "Date"),
            Self::Time => write!(f, "Time"),
            Self::Duration => write!(f, "Duration"),
            Self::Any => write!(f, "Any"),
            Self::Union(types) => {
                let mut iter = types.iter();
                if let Some(first) = iter.next() {
                    write!(f, "{first}")?;
                }
                for _ in 0..types.len().saturating_sub(2) {
                    if let Some(next) = iter.next() {
                        write!(f, ", {next}")?;
                    }
                }
                if let Some(last) = iter.next() {
                    if types.len() > 2 {
                        write!(f, ",")?;
                    }
                    write!(f, " or {last}")?;
                }
                Ok(())
            }
            Self::Optional(inner) => write!(f, "{inner}"),
        }
    }
}

#[derive(Debug)]
pub enum FnArguments {
    Fixed(Vec<Type>),
    VarLength(Type),
}

pub struct GraphFn {
    pub name: String,
    pub func: RuntimeFn,
    pub write: bool,
    pub non_deterministic: bool,
    /// Optional pure form of the function: when set, the optimizer's
    /// constant-folding pass invokes this directly with concrete arguments
    /// without needing a [`Runtime`]. Used for Map/String constructor forms
    /// (e.g. `date('2020-01-01')`). The binder only invokes `pure_fn` when
    /// the arguments match `pure_args_type` — this matters because
    /// constructor functions are registered with a loose `var_arg` signature
    /// to accept zero-arg "now"-style invocations, while `pure_fn` only
    /// handles the single-argument constructor shape.
    pub pure_fn: Option<fn(&[Value]) -> Result<Value, String>>,
    /// Argument types `pure_fn` accepts (positional, one entry per arg).
    /// Empty when `pure_fn` is `None`.
    pub pure_args_type: Vec<Type>,
    /// Optional positional-slot form used by the binder's struct-constructor
    /// rewrite: a call like `duration({months: i})` is rewritten into 7
    /// positional children, which eval.rs evaluates into a stack-allocated
    /// `[Value; 10]` array and dispatches here — bypassing the
    /// `ThinVec<Value>` heap alloc, `validate_args_type` walk, and the
    /// `RuntimeFn` dispatch hop used by the general `func` path. Only
    /// invoked when `num_children > 1`.
    pub struct_fn: Option<fn(&[Value]) -> Result<Value, String>>,
    /// Slot names that drive the binder rewrite — the order matches the
    /// positional children produced for `struct_fn`. Set in lockstep with
    /// `struct_fn`; both `Some` or both `None`.
    pub struct_slots: &'static [&'static str],
    pub args_type: FnArguments,
    pub fn_type: FnType,
    pub ret_type: Type,
}

impl Debug for GraphFn {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("GraphFn")
            .field("name", &self.name)
            .field("write", &self.write)
            .field("non_deterministic", &self.non_deterministic)
            .field("args_type", &self.args_type)
            .field("fn_type", &self.fn_type)
            .field("ret_type", &self.ret_type)
            .finish_non_exhaustive()
    }
}

impl GraphFn {
    #[must_use]
    pub fn new(
        name: &str,
        func: fn(&Runtime, &[Value]) -> Result<Value, String>,
        write: bool,
        non_deterministic: bool,
        args_type: FnArguments,
        fn_type: FnType,
        ret_type: Type,
    ) -> Self {
        Self {
            name: String::from(name),
            func: RuntimeFn::Native(func),
            write,
            non_deterministic,
            pure_fn: None,
            pure_args_type: Vec::new(),
            struct_fn: None,
            struct_slots: &[],
            args_type,
            fn_type,
            ret_type,
        }
    }

    #[must_use]
    pub fn new_procedure(
        name: &str,
        func: fn(&Runtime, &[Value]) -> Result<ProcedureBatch, String>,
        write: bool,
        non_deterministic: bool,
        args_type: FnArguments,
        fn_type: FnType,
        ret_type: Type,
    ) -> Self {
        Self {
            name: String::from(name),
            func: RuntimeFn::NativeProcedureBatch(func),
            write,
            non_deterministic,
            pure_fn: None,
            pure_args_type: Vec::new(),
            struct_fn: None,
            struct_slots: &[],
            args_type,
            fn_type,
            ret_type,
        }
    }

    #[must_use]
    pub fn new_udf(name: &str) -> Self {
        let udf_name = name.to_string();
        Self {
            name: String::from(name),
            func: RuntimeFn::Udf(udf_name),
            write: false,
            non_deterministic: false,
            pure_fn: None,
            pure_args_type: Vec::new(),
            struct_fn: None,
            struct_slots: &[],
            args_type: FnArguments::VarLength(Type::Any),
            fn_type: FnType::Udf,
            ret_type: Type::Any,
        }
    }

    #[must_use]
    pub const fn is_aggregate(&self) -> bool {
        matches!(self.fn_type, FnType::Aggregation { .. })
    }

    pub fn validate(
        &self,
        args: usize,
    ) -> Result<(), String> {
        match &self.args_type {
            FnArguments::Fixed(args_type) => {
                let least = args_type
                    .iter()
                    .filter(|x| !matches!(x, Type::Optional(_)))
                    .count();
                if args < least {
                    return Err(format!(
                        "Received {args} arguments to function '{}', expected at least {least}",
                        self.name
                    ));
                }
                let most = args_type.len();
                if args > most {
                    return Err(format!(
                        "Received {} arguments to function '{}', expected at most {}",
                        args,
                        self.name,
                        args_type.len()
                    ));
                }
            }
            FnArguments::VarLength(_) => {}
        }
        Ok(())
    }
}

impl GraphFn {
    pub fn validate_args_type<V: Borrow<Value>>(
        &self,
        args: &[V],
    ) -> Result<(), String> {
        match &self.args_type {
            FnArguments::Fixed(args_type) => {
                for (i, arg_type) in args_type.iter().enumerate() {
                    if i >= args.len() {
                        if !matches!(arg_type, Type::Optional(_)) {
                            return Err(format!(
                                "Missing argument {} for function '{}', expected type {:?}",
                                i + 1,
                                self.name,
                                arg_type
                            ));
                        }
                    } else if let Some((actual, expected)) =
                        args[i].borrow().value_of_type(arg_type)
                    {
                        return Err(format!(
                            "Type mismatch: expected {expected} but was {actual}"
                        ));
                    }
                }
            }
            FnArguments::VarLength(_) => {}
        }
        Ok(())
    }
}

impl GraphFn {
    /// Validates domain constraints (e.g., percentile must be in [0.0, 1.0])
    /// This is called AFTER type validation but BEFORE consuming the accumulator
    pub fn validate_args_domain(
        &self,
        args: &[Value],
    ) -> Result<(), String> {
        // Only percentile functions need domain validation currently.
        // Allocation-free case-insensitive prefix check: this runs per row
        // in aggregation hot loops, where `to_lowercase` showed up in
        // profiles as pure alloc churn.
        if self
            .name
            .get(.."percentile".len())
            .is_some_and(|p| p.eq_ignore_ascii_case("percentile"))
        {
            // percentile is at index 1 (after the value argument)
            if args.len() >= 2 {
                if matches!(args[1], Value::Null) {
                    return Err("Type mismatch: expected Integer or Float but was Null".to_string());
                }
                let percentile = args[1].get_numeric();
                if !(0.0..=1.0).contains(&percentile) {
                    return Err(format!(
                        "Invalid input - '{percentile}' is not a valid argument, must be a number in the range 0.0 to 1.0"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Execute a procedure and return its native columnar batch output.
    pub fn call_procedure_batch(
        &self,
        rt: &Runtime,
        args: &[Value],
    ) -> Result<ProcedureBatch, String> {
        if !matches!(self.fn_type, FnType::Procedure(_)) {
            return Err(format!("Function '{}' is not a procedure", self.name));
        }
        self.func.call_procedure_batch(rt, args)
    }
}

#[derive(Default, Debug)]
pub struct Functions {
    functions: HashMap<String, Arc<GraphFn>>,
}

#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Functions {}
unsafe impl Sync for Functions {}

impl Functions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        name: &str,
        func: fn(&Runtime, &[Value]) -> Result<Value, String>,
        write: bool,
        non_deterministic: bool,
        args_type: Vec<Type>,
        fn_type: FnType,
        ret_type: Type,
    ) {
        let lower_name = name.to_lowercase();
        assert!(
            !self.functions.contains_key(&lower_name),
            "Function '{name}' already exists"
        );
        let graph_fn = Arc::new(GraphFn::new(
            name,
            func,
            write,
            non_deterministic,
            FnArguments::Fixed(args_type),
            fn_type,
            ret_type,
        ));
        self.functions.insert(lower_name, graph_fn);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_procedure(
        &mut self,
        name: &str,
        func: fn(&Runtime, &[Value]) -> Result<ProcedureBatch, String>,
        write: bool,
        non_deterministic: bool,
        args_type: Vec<Type>,
        fn_type: FnType,
        ret_type: Type,
    ) {
        let lower_name = name.to_lowercase();
        assert!(
            !self.functions.contains_key(&lower_name),
            "Function '{name}' already exists"
        );
        let graph_fn = Arc::new(GraphFn::new_procedure(
            name,
            func,
            write,
            non_deterministic,
            FnArguments::Fixed(args_type),
            fn_type,
            ret_type,
        ));
        self.functions.insert(lower_name, graph_fn);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_var_len(
        &mut self,
        name: &str,
        func: fn(&Runtime, &[Value]) -> Result<Value, String>,
        write: bool,
        non_deterministic: bool,
        arg_type: Type,
        fn_type: FnType,
        ret_type: Type,
    ) {
        let name = name.to_lowercase();
        assert!(
            !self.functions.contains_key(&name),
            "Function '{name}' already exists"
        );
        let graph_fn = Arc::new(GraphFn::new(
            &name,
            func,
            write,
            non_deterministic,
            FnArguments::VarLength(arg_type),
            fn_type,
            ret_type,
        ));
        self.functions.insert(name, graph_fn);
    }

    /// Attach a pure form of a previously registered function. The binder's
    /// constant-folding pass invokes `pure_fn` when all arguments are
    /// constants and match `args_type` positionally.
    pub fn set_pure_fn(
        &mut self,
        name: &str,
        pure_fn: fn(&[Value]) -> Result<Value, String>,
        args_type: Vec<Type>,
    ) {
        let lower = name.to_lowercase();
        let existing = self
            .functions
            .get_mut(&lower)
            .unwrap_or_else(|| panic!("Function '{name}' not registered"));
        let inner =
            Arc::get_mut(existing).expect("function Arc must be unique at registration time");
        inner.pure_fn = Some(pure_fn);
        inner.pure_args_type = args_type;
    }

    /// Attach a positional-slot constructor to a previously registered
    /// function. The binder rewrites Map-keyed constructor calls like
    /// `duration({months: i})` into positional children using `slots` for
    /// the slot order, and eval.rs dispatches them through `struct_fn` via
    /// a stack `[Value; 10]` array.
    pub fn set_struct_fn(
        &mut self,
        name: &str,
        struct_fn: fn(&[Value]) -> Result<Value, String>,
        slots: &'static [&'static str],
    ) {
        let lower = name.to_lowercase();
        let existing = self
            .functions
            .get_mut(&lower)
            .unwrap_or_else(|| panic!("Function '{name}' not registered"));
        let inner =
            Arc::get_mut(existing).expect("function Arc must be unique at registration time");
        inner.struct_fn = Some(struct_fn);
        inner.struct_slots = slots;
    }

    pub fn get(
        &self,
        name: &str,
        fn_type: &FnType,
    ) -> Result<Arc<GraphFn>, String> {
        let lower = name.to_lowercase();
        // Try built-in functions first
        if let Some(graph_fn) = self.functions.get(lower.as_str())
            && &graph_fn.fn_type == fn_type
        {
            return Ok(graph_fn.clone());
        }
        // Fall back to dynamic UDF registry (only for scalar/UDF function lookups,
        // not for Procedure or Aggregation which have distinct semantics).
        if matches!(fn_type, FnType::Function | FnType::Udf)
            && let Some(reg) = UDF_FUNCTIONS.get()
        {
            let guard = reg.read();
            if let Some(graph_fn) = guard.get(lower.as_str()) {
                return Ok(graph_fn.clone());
            }
        }
        Err(format!("Unknown function '{name}'"))
    }

    /// Iterate over all registered functions.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<GraphFn>> {
        self.functions.values()
    }

    #[must_use]
    pub fn is_aggregate(
        &self,
        name: &str,
    ) -> bool {
        self.functions
            .get(name)
            .is_some_and(|graph_fn| matches!(graph_fn.fn_type, FnType::Aggregation { .. }))
    }
}

static FUNCTIONS: OnceLock<Functions> = OnceLock::new();

/// Dynamic UDF function registry, separate from built-in functions.
static UDF_FUNCTIONS: OnceLock<RwLock<HashMap<String, Arc<GraphFn>>>> = OnceLock::new();

/// Global version counter for UDF changes. Incremented on register/unregister/flush.
/// Used to invalidate query plan caches.
static UDF_VERSION: AtomicU64 = AtomicU64::new(0);

/// Get the current UDF version (for plan cache invalidation).
pub fn udf_version() -> u64 {
    UDF_VERSION.load(Ordering::Acquire)
}

pub fn init_functions() -> Result<(), Functions> {
    let mut funcs = Functions::new();

    entity::register(&mut funcs);
    string::register(&mut funcs);
    math::register(&mut funcs);
    trig::register(&mut funcs);
    conversion::register(&mut funcs);
    list::register(&mut funcs);
    aggregation::register(&mut funcs);
    spatial::register(&mut funcs);
    temporal::register(&mut funcs);
    path::register(&mut funcs);
    internal::register(&mut funcs);
    procedures::register(&mut funcs);
    algo_procedures::register(&mut funcs);

    FUNCTIONS.set(funcs)
}

/// Initialize the dynamic UDF function registry.
pub fn init_udf_functions() {
    let _ = UDF_FUNCTIONS.set(RwLock::new(HashMap::new()));
}

/// Register a UDF in the dynamic registry.
pub fn register_udf(
    name: &str,
    func: Arc<GraphFn>,
) {
    if let Some(reg) = UDF_FUNCTIONS.get() {
        reg.write().insert(name.to_lowercase(), func);
        UDF_VERSION.fetch_add(1, Ordering::Release);
    }
}

/// Unregister a UDF from the dynamic registry.
pub fn unregister_udf(name: &str) {
    if let Some(reg) = UDF_FUNCTIONS.get() {
        reg.write().remove(&name.to_lowercase());
        UDF_VERSION.fetch_add(1, Ordering::Release);
    }
}

/// Remove all UDFs from the dynamic registry.
pub fn flush_udfs() {
    if let Some(reg) = UDF_FUNCTIONS.get() {
        reg.write().clear();
        UDF_VERSION.fetch_add(1, Ordering::Release);
    }
}

/// Snapshot the current UDF registry entries.
/// Returns an empty Vec if the registry is not yet initialized.
pub fn get_udf_functions() -> Vec<Arc<GraphFn>> {
    UDF_FUNCTIONS
        .get()
        .map(|reg| reg.read().values().cloned().collect())
        .unwrap_or_default()
}

pub fn get_functions() -> &'static Functions {
    FUNCTIONS.get().expect("Functions not initialized")
}
