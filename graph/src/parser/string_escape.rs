//! Cypher-compliant string escape and unescape functions.
//!
//! This module provides functions to handle escape sequences in Cypher string
//! literals, maintaining strict behavioral parity with the C reference
//! implementation in FalkorDB.
//!
//! ## Usage in the Parser
//!
//! The lexer ([`crate::parser::lexer`]) calls [`cypher_unescape`] when it
//! encounters a string literal token (single- or double-quoted). The result
//! is stored as the unescaped `Token::String` value in the token stream.
//!
//! The reverse function [`cypher_escape`] is used when serializing values
//! back to Cypher-safe string representations (e.g., in EXPLAIN output or
//! error messages).
//!
//! ## Supported Escape Sequences
//!
//! ```text
//!  Sequence   Character        ASCII
//!  --------   ---------        -----
//!  \a         bell/alert       0x07
//!  \b         backspace        0x08
//!  \f         form feed        0x0C
//!  \n         newline          0x0A
//!  \r         carriage return  0x0D
//!  \t         horizontal tab   0x09
//!  \v         vertical tab     0x0B
//!  \\         backslash        0x5C
//!  \'         single quote     0x27
//!  \"         double quote     0x22
//!  \?         question mark    0x3F
//! ```
//!
//! Unrecognized escape sequences (e.g., `\x`, `\z`) are preserved as-is:
//! both the backslash and the following character are kept in the output.

/// Unescapes a Cypher string literal.
///
/// Converts escape sequences to their actual character representations according to Cypher rules:
/// - `\a` → bell/alert (ASCII 0x07)
/// - `\b` → backspace (ASCII 0x08)
/// - `\f` → form feed (ASCII 0x0C)
/// - `\n` → newline (ASCII 0x0A)
/// - `\r` → carriage return (ASCII 0x0D)
/// - `\t` → horizontal tab (ASCII 0x09)
/// - `\v` → vertical tab (ASCII 0x0B)
/// - `\\` → backslash
/// - `\'` → single quote
/// - `\"` → double quote
/// - `\?` → question mark
///
/// Unrecognized escape sequences (e.g., `\x`, `\z`) are preserved as-is (both backslash and character).
///
/// # Arguments
/// * `input` - The raw string slice containing escape sequences
///
/// # Returns
/// * `Ok(String)` - The unescaped string
/// * `Err(String)` - Error message if the string ends with an incomplete escape sequence
///
/// # Examples
/// ```
/// use graph::parser::string_escape::cypher_unescape;
///
/// assert_eq!(cypher_unescape(r"hello\nworld").unwrap(), "hello\nworld");
/// assert_eq!(cypher_unescape(r"single \' char").unwrap(), "single ' char");
/// assert_eq!(cypher_unescape(r#"double \" char"#).unwrap(), "double \" char");
/// assert_eq!(cypher_unescape(r"backslash \\").unwrap(), "backslash \\");
/// ```
pub fn cypher_unescape(input: &str) -> Result<String, String> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('a') => result.push('\u{0007}'), // Bell/alert
                Some('b') => result.push('\u{0008}'), // Backspace
                Some('f') => result.push('\u{000C}'), // Form feed
                Some('n') => result.push('\n'),       // Newline
                Some('r') => result.push('\r'),       // Carriage return
                Some('t') => result.push('\t'),       // Horizontal tab
                Some('v') => result.push('\u{000B}'), // Vertical tab
                Some('\\') => result.push('\\'),      // Backslash
                Some('\'') => result.push('\''),      // Single quote
                Some('"') => result.push('"'),        // Double quote
                Some('?') => result.push('?'),        // Question mark
                Some('u') => {
                    // \uXXXX - 4-digit hex Unicode escape
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() != 4 {
                        return Err(format!("Invalid unicode escape: \\u{hex}"));
                    }
                    let code = u32::from_str_radix(&hex, 16)
                        .map_err(|_| format!("Invalid unicode escape: \\u{hex}"))?;
                    let c = char::from_u32(code)
                        .ok_or_else(|| format!("Invalid unicode code point: \\u{hex}"))?;
                    result.push(c);
                }
                Some('U') => {
                    // \UXXXXXXXX - 8-digit hex Unicode escape
                    let hex: String = chars.by_ref().take(8).collect();
                    if hex.len() != 8 {
                        return Err(format!("Invalid unicode escape: \\U{hex}"));
                    }
                    let code = u32::from_str_radix(&hex, 16)
                        .map_err(|_| format!("Invalid unicode escape: \\U{hex}"))?;
                    let c = char::from_u32(code)
                        .ok_or_else(|| format!("Invalid unicode code point: \\U{hex}"))?;
                    result.push(c);
                }
                Some(other) => {
                    // Unrecognized escape sequence - keep as-is
                    result.push('\\');
                    result.push(other);
                }
                None => {
                    // String ends with backslash
                    return Err("Unterminated escape sequence at end of string".to_string());
                }
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

/// Escapes a Cypher string for output.
///
/// Converts special characters to their escape sequence representations:
/// - bell/alert → `\a`
/// - backspace → `\b`
/// - form feed → `\f`
/// - newline → `\n`
/// - carriage return → `\r`
/// - horizontal tab → `\t`
/// - vertical tab → `\v`
/// - backslash → `\\`
/// - single quote → `\'`
/// - double quote → `\"`
///
/// # Arguments
/// * `input` - The string to escape
///
/// # Returns
/// * `String` - The escaped string
///
/// # Examples
/// ```
/// use graph::parser::string_escape::cypher_escape;
///
/// assert_eq!(cypher_escape("hello\nworld"), r"hello\nworld");
/// assert_eq!(cypher_escape("single ' char"), r"single \' char");
/// assert_eq!(cypher_escape("double \" char"), r#"double \" char"#);
/// assert_eq!(cypher_escape("backslash \\"), r"backslash \\");
/// ```
#[must_use]
pub fn cypher_escape(input: &str) -> String {
    let mut result = String::with_capacity(input.len() * 2);

    for ch in input.chars() {
        match ch {
            '\u{0007}' => result.push_str(r"\a"),
            '\u{0008}' => result.push_str(r"\b"),
            '\u{000C}' => result.push_str(r"\f"),
            '\n' => result.push_str(r"\n"),
            '\r' => result.push_str(r"\r"),
            '\t' => result.push_str(r"\t"),
            '\u{000B}' => result.push_str(r"\v"),
            '\\' => result.push_str(r"\\"),
            '\'' => result.push_str(r"\'"),
            '"' => result.push_str(r#"\""#),
            _ => result.push(ch),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Unescape Tests
    // ========================================================================

    #[test]
    fn test_unescape_standard_escapes() {
        // Test all standard escape sequences
        assert_eq!(cypher_unescape(r"\a").unwrap(), "\u{0007}");
        assert_eq!(cypher_unescape(r"\b").unwrap(), "\u{0008}");
        assert_eq!(cypher_unescape(r"\f").unwrap(), "\u{000C}");
        assert_eq!(cypher_unescape(r"\n").unwrap(), "\n");
        assert_eq!(cypher_unescape(r"\r").unwrap(), "\r");
        assert_eq!(cypher_unescape(r"\t").unwrap(), "\t");
        assert_eq!(cypher_unescape(r"\v").unwrap(), "\u{000B}");
        assert_eq!(cypher_unescape(r"\\").unwrap(), "\\");
        assert_eq!(cypher_unescape(r"\'").unwrap(), "'");
        assert_eq!(cypher_unescape(r#"\""#).unwrap(), "\"");
        assert_eq!(cypher_unescape(r"\?").unwrap(), "?");
    }

    #[test]
    fn test_unescape_single_quote() {
        // Test case from the failing test
        assert_eq!(cypher_unescape(r"single \' char").unwrap(), "single ' char");
    }

    #[test]
    fn test_unescape_double_quote() {
        // Test case from the failing test
        assert_eq!(
            cypher_unescape(r#"double \" char"#).unwrap(),
            "double \" char"
        );
    }

    #[test]
    fn test_unescape_mixed_quotes() {
        // Test case from the failing test
        assert_eq!(
            cypher_unescape(r#"mixed \' and \" chars"#).unwrap(),
            "mixed ' and \" chars"
        );
    }

    #[test]
    fn test_unescape_multiple_escapes() {
        assert_eq!(
            cypher_unescape(r"Tab\tThen\!Now\n").unwrap(),
            "Tab\tThen\\!Now\n"
        );
        assert_eq!(
            cypher_unescape(r"Mix\\path\nof\xfiles").unwrap(),
            "Mix\\path\nof\\xfiles"
        );
    }

    #[test]
    fn test_unescape_quotes_escaped() {
        assert_eq!(
            cypher_unescape(r#"Quotes: \"hello\" and \'bye\'"#).unwrap(),
            "Quotes: \"hello\" and 'bye'"
        );
    }

    #[test]
    fn test_unescape_start_and_end() {
        assert_eq!(
            cypher_unescape(r"\nStart and end\t").unwrap(),
            "\nStart and end\t"
        );
    }

    #[test]
    fn test_unescape_unrecognized_escapes_preserved() {
        // Unrecognized escape sequences should be kept as-is
        assert_eq!(cypher_unescape(r"\x").unwrap(), r"\x");
        assert_eq!(cypher_unescape(r"\z").unwrap(), r"\z");
        assert_eq!(cypher_unescape(r"\!").unwrap(), r"\!");
        assert_eq!(cypher_unescape(r"\1").unwrap(), r"\1");
        assert_eq!(cypher_unescape(r"\@").unwrap(), r"\@");
    }

    #[test]
    fn test_unescape_empty_string() {
        assert_eq!(cypher_unescape("").unwrap(), "");
    }

    #[test]
    fn test_unescape_no_escapes() {
        assert_eq!(cypher_unescape("hello world").unwrap(), "hello world");
        assert_eq!(cypher_unescape("simple text").unwrap(), "simple text");
    }

    #[test]
    fn test_unescape_consecutive_escapes() {
        assert_eq!(cypher_unescape(r"\\\\").unwrap(), "\\\\");
        assert_eq!(cypher_unescape(r"\n\n\n").unwrap(), "\n\n\n");
        assert_eq!(cypher_unescape(r"\'\'\'").unwrap(), "'''");
    }

    #[test]
    fn test_unescape_mixed_content() {
        assert_eq!(
            cypher_unescape(r"Line1\nLine2\tTabbed\rReturn").unwrap(),
            "Line1\nLine2\tTabbed\rReturn"
        );
    }

    #[test]
    fn test_unescape_unicode_preserved() {
        // Unicode characters should pass through unchanged
        assert_eq!(cypher_unescape("hello 世界").unwrap(), "hello 世界");
        assert_eq!(cypher_unescape("emoji 🚀🔥").unwrap(), "emoji 🚀🔥");
        assert_eq!(cypher_unescape(r"🧐🍌❖⋙⚐").unwrap(), "🧐🍌❖⋙⚐");
    }

    #[test]
    fn test_unescape_error_unterminated() {
        // String ending with backslash should error
        let result = cypher_unescape(r"incomplete\");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Unterminated escape sequence at end of string"
        );
    }

    #[test]
    fn test_unescape_backslash_variations() {
        // Test unrecognized escape sequences that should be preserved as backslash + char
        assert_eq!(cypher_unescape(r"one\xwo").unwrap(), "one\\xwo"); // \x is unrecognized
        assert_eq!(cypher_unescape(r"path\zo\desk").unwrap(), "path\\zo\\desk"); // \z and \d are unrecognized
        assert_eq!(
            cypher_unescape(r"C:\\Windows\\System32").unwrap(),
            "C:\\Windows\\System32"
        ); // \\ is backslash
    }

    #[test]
    fn test_unescape_all_escapes_in_one() {
        // Test all standard escape sequences in one string
        let input = "\\a\\b\\f\\n\\r\\t\\v\\\\\\'\\\"";
        // Build expected string: bell, backspace, formfeed, newline, return, tab, vtab, backslash, quote, doublequote
        let mut expected = String::new();
        expected.push('\u{0007}'); // bell
        expected.push('\u{0008}'); // backspace
        expected.push('\u{000C}'); // formfeed
        expected.push('\n'); // newline
        expected.push('\r'); // carriage return
        expected.push('\t'); // tab
        expected.push('\u{000B}'); // vertical tab
        expected.push('\\'); // backslash
        expected.push('\''); // single quote
        expected.push('"'); // double quote
        assert_eq!(cypher_unescape(input).unwrap(), expected);
    }

    // ========================================================================
    // Escape Tests
    // ========================================================================

    #[test]
    fn test_escape_standard_characters() {
        assert_eq!(cypher_escape("\u{0007}"), r"\a");
        assert_eq!(cypher_escape("\u{0008}"), r"\b");
        assert_eq!(cypher_escape("\u{000C}"), r"\f");
        assert_eq!(cypher_escape("\n"), r"\n");
        assert_eq!(cypher_escape("\r"), r"\r");
        assert_eq!(cypher_escape("\t"), r"\t");
        assert_eq!(cypher_escape("\u{000B}"), r"\v");
        assert_eq!(cypher_escape("\\"), r"\\");
        assert_eq!(cypher_escape("'"), r"\'");
        assert_eq!(cypher_escape("\""), r#"\""#);
    }

    #[test]
    fn test_escape_single_quote() {
        assert_eq!(cypher_escape("single ' char"), r"single \' char");
    }

    #[test]
    fn test_escape_double_quote() {
        assert_eq!(cypher_escape("double \" char"), r#"double \" char"#);
    }

    #[test]
    fn test_escape_mixed_quotes() {
        assert_eq!(
            cypher_escape("mixed ' and \" chars"),
            r#"mixed \' and \" chars"#
        );
    }

    #[test]
    fn test_escape_newlines_and_tabs() {
        assert_eq!(cypher_escape("hello\nworld"), r"hello\nworld");
        assert_eq!(cypher_escape("tab\there"), r"tab\there");
        assert_eq!(
            cypher_escape("Line1\nLine2\tTabbed\rReturn"),
            r"Line1\nLine2\tTabbed\rReturn"
        );
    }

    #[test]
    fn test_escape_backslashes() {
        assert_eq!(cypher_escape("backslash \\"), r"backslash \\");
        assert_eq!(cypher_escape("\\\\"), r"\\\\");
        assert_eq!(
            cypher_escape("C:\\Windows\\System32"),
            r"C:\\Windows\\System32"
        );
    }

    #[test]
    fn test_escape_empty_string() {
        assert_eq!(cypher_escape(""), "");
    }

    #[test]
    fn test_escape_no_special_chars() {
        assert_eq!(cypher_escape("hello world"), "hello world");
        assert_eq!(cypher_escape("simple text"), "simple text");
    }

    #[test]
    fn test_escape_unicode_preserved() {
        // Unicode characters should pass through unchanged
        assert_eq!(cypher_escape("hello 世界"), "hello 世界");
        assert_eq!(cypher_escape("emoji 🚀🔥"), "emoji 🚀🔥");
        assert_eq!(cypher_escape("🧐🍌❖⋙⚐"), "🧐🍌❖⋙⚐");
    }

    #[test]
    fn test_escape_all_special_chars() {
        // Build the input string with actual special characters
        let mut input = String::new();
        input.push('\u{0007}'); // bell
        input.push('\u{0008}'); // backspace
        input.push('\u{000C}'); // formfeed
        input.push('\n'); // newline
        input.push('\r'); // carriage return
        input.push('\t'); // tab
        input.push('\u{000B}'); // vertical tab
        input.push('\\'); // backslash
        input.push('\''); // single quote
        input.push('"'); // double quote
        let expected = "\\a\\b\\f\\n\\r\\t\\v\\\\\\'\\\"";
        assert_eq!(cypher_escape(&input), expected);
    }

    // ========================================================================
    // Round-trip Tests (escape then unescape should give original)
    // ========================================================================

    #[test]
    fn test_roundtrip_simple() {
        let original = "hello\nworld\ttab";
        let escaped = cypher_escape(original);
        let unescaped = cypher_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_roundtrip_quotes() {
        let original = "single ' and double \"";
        let escaped = cypher_escape(original);
        let unescaped = cypher_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_roundtrip_backslashes() {
        let original = "path\\to\\file";
        let escaped = cypher_escape(original);
        let unescaped = cypher_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_roundtrip_all_special_chars() {
        let original = format!(
            "{}{}{}\n\r\t{}\\'{}",
            '\u{0007}', '\u{0008}', '\u{000C}', '\u{000B}', '"'
        );
        let escaped = cypher_escape(&original);
        let unescaped = cypher_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_roundtrip_unicode() {
        let original = "hello 世界 🚀 with\nnewlines";
        let escaped = cypher_escape(original);
        let unescaped = cypher_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    // ========================================================================
    // Test cases from the C reference (test_params.py)
    // ========================================================================

    #[test]
    fn test_c_reference_parity_valid_escapes() {
        // From test_params.py in C reference
        assert_eq!(cypher_unescape(r"\a").unwrap(), "\u{0007}");
        assert_eq!(cypher_unescape(r"\b").unwrap(), "\u{0008}");
        assert_eq!(cypher_unescape(r"\f").unwrap(), "\u{000C}");
        assert_eq!(cypher_unescape(r"\n").unwrap(), "\n");
        assert_eq!(cypher_unescape(r"\r").unwrap(), "\r");
        assert_eq!(cypher_unescape(r"\t").unwrap(), "\t");
        assert_eq!(cypher_unescape(r"\v").unwrap(), "\u{000B}");
        assert_eq!(cypher_unescape(r"\\").unwrap(), "\\");
    }

    #[test]
    fn test_c_reference_parity_complex() {
        assert_eq!(
            cypher_unescape(r"Tab\tThen\!Now\n").unwrap(),
            "Tab\tThen\\!Now\n"
        );
        assert_eq!(
            cypher_unescape(r"Mix\\path\nof\xfiles").unwrap(),
            "Mix\\path\nof\\xfiles"
        );
        assert_eq!(
            cypher_unescape(r#"Quotes: \"hello\" and \'bye\'"#).unwrap(),
            "Quotes: \"hello\" and 'bye'"
        );
        assert_eq!(
            cypher_unescape(r"\nStart and end\t").unwrap(),
            "\nStart and end\t"
        );
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_edge_case_only_backslashes() {
        assert_eq!(cypher_unescape(r"\\").unwrap(), "\\");
        assert_eq!(cypher_unescape(r"\\\\").unwrap(), "\\\\");
        assert_eq!(cypher_unescape(r"\\\\\\\\").unwrap(), "\\\\\\\\");
    }

    #[test]
    fn test_edge_case_alternating() {
        assert_eq!(cypher_unescape(r"a\nb\tc").unwrap(), "a\nb\tc");
        assert_eq!(cypher_unescape(r"\na\nb\n").unwrap(), "\na\nb\n");
    }

    #[test]
    fn test_edge_case_long_string() {
        let input = "This is a very long string with many \\n newlines \\t tabs \\\\ backslashes \\' quotes \\\" and more \\n\\n\\n";
        let expected = "This is a very long string with many \n newlines \t tabs \\ backslashes ' quotes \" and more \n\n\n";
        assert_eq!(cypher_unescape(input).unwrap(), expected);
    }

    #[test]
    fn test_edge_case_only_escapes() {
        assert_eq!(cypher_unescape(r"\n\t\r").unwrap(), "\n\t\r");
    }

    #[test]
    fn test_edge_case_unrecognized_at_end() {
        assert_eq!(cypher_unescape(r"text\x").unwrap(), "text\\x");
    }
}
