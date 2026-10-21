//! Upper bound on the length of every name the engine accepts.
//!
//! Labels, relationship types, property keys, aliases, procedure and
//! function names, UDF library names — anything a query or command can
//! introduce as an identifier — are capped at [`MAX_IDENTIFIER_LEN`].
//! Mirrors `src/util/identifier_limits.h` in FalkorDB C so both engines
//! reject the same inputs with the same message.

/// Maximum length, in bytes, of an identifier.
pub const MAX_IDENTIFIER_LEN: usize = 512;

/// Tail of every message [`validate_identifier_len`] produces, and what
/// [`is_identifier_too_long`] recognises. Spelled out rather than formatted so
/// it can be matched against; the assertion below keeps it honest.
const TOO_LONG: &str = "exceeds maximum length of 512 characters";
const _: () = assert!(
    MAX_IDENTIFIER_LEN == 512,
    "TOO_LONG must spell out MAX_IDENTIFIER_LEN"
);

/// Reject `name` when it is longer than [`MAX_IDENTIFIER_LEN`].
///
/// `entity` describes the kind of identifier being checked and opens the
/// error message, e.g. `"Label name"` produces
/// `"Label name exceeds maximum length of 512 characters"`.
pub fn validate_identifier_len(
    name: &str,
    entity: &str,
) -> Result<(), String> {
    if name.len() > MAX_IDENTIFIER_LEN {
        return Err(format!("{entity} {TOO_LONG}"));
    }
    Ok(())
}

/// Whether `err` is a rejection produced by [`validate_identifier_len`].
///
/// Lets a caller that would otherwise replace a nested failure with its own
/// summary pass this one through: it names a definite input the user has to
/// shorten, which a generic "could not parse" message would hide.
#[must_use]
pub fn is_identifier_too_long(err: &str) -> bool {
    err.ends_with(TOO_LONG)
}
