//! Cypher query parsing.
//!
//! This module contains a hand-written recursive descent parser for the
//! Cypher query language (not generated from a grammar file). It converts
//! raw query strings into an abstract syntax tree (AST) that downstream
//! stages -- binder, planner, optimizer, and runtime -- consume.
//!
//! ## Parsing Pipeline
//!
//! ```text
//!  "MATCH (n:Person)-[r:KNOWS]->(m) WHERE n.age > 30 RETURN m.name"
//!                            |
//!                            v
//!                   +----------------+
//!                   |     Lexer      |  lexer.rs
//!                   | (tokenization) |  string_escape.rs
//!                   +----------------+
//!                            |
//!          [MATCH, LPAREN, IDENT("n"), COLON, IDENT("Person"), ...]
//!                            |
//!                            v
//!                   +----------------+
//!                   |     Parser     |  cypher.rs
//!                   | (recursive     |  macro.rs
//!                   |  descent)      |
//!                   +----------------+
//!                            |
//!                            v
//!                  RawQueryIR (ast.rs)
//!                  = QueryIR<Arc<String>>
//! ```
//!
//! The output is a `RawQueryIR` -- a `QueryIR` tree parameterized by raw
//! string variable names (`Arc<String>`). After binding, variables are
//! resolved to numeric IDs and the tree becomes a `BoundQueryIR`
//! (`QueryIR<Variable>`).
//!
//! ## Submodules
//!
//! - [`ast`]: AST node definitions (`ExprIR`, `QueryIR`, `QueryGraph`, etc.)
//! - [`lexer`]: Tokenizer producing `Token` variants from a query string
//! - [`cypher`]: Recursive descent parser building the AST from tokens
//! - [`r#macro`]: Internal macros for token matching and operator folding
//! - [`string_escape`]: Cypher string literal escape/unescape utilities

pub mod ast;
#[macro_use]
pub mod r#macro;
pub mod cypher;
pub mod lexer;
pub mod string_escape;
