//! Cypher query parser for FalkorDB.
//!
//! This module implements a hand-written recursive descent parser for the
//! Cypher query language. It converts Cypher query strings into an Abstract
//! Syntax Tree (AST) defined in [`crate::parser::ast`].
//! Lexical analysis is provided by [`crate::parser::lexer`], and parser
//! helper macros are defined in [`crate::parser::r#macro`].
//!
//! ## Architecture
//!
//! The [`Parser`] struct wraps a [`crate::parser::lexer::Lexer`] and maintains
//! a counter for generating anonymous variable names. Parsing proceeds by
//! recursive descent: each grammar rule maps to a `parse_*` method.
//!
//! ## Entry Point
//!
//! The main entry point is [`parse`], which takes a query string and returns
//! a [`RawQueryIR`] (unbound AST with `Arc<String>` variable names).
//!
//! ```text
//! "MATCH (n) RETURN n"
//!         |
//!         v
//!     Lexer --> [MATCH, LPAREN, IDENT("n"), RPAREN, RETURN, IDENT("n"), EOF]
//!         |
//!         v
//!     Parser --> QueryIR::Query {
//!                  clauses: [
//!                    QueryIR::Match { pattern: ..., filter: None, optional: false },
//!                    QueryIR::Return { exprs: [("n", var("n"))], ... }
//!                  ],
//!                  write: false
//!                }
//! ```
//!
//! ## Grammar Structure
//!
//! The parser handles the following Cypher grammar (simplified):
//!
//! ```text
//! query           ::= single_query ( UNION [ALL] single_query )*
//! single_query    ::= reading_clause* writing_clause*
//!
//! reading_clause  ::= MATCH | OPTIONAL MATCH | UNWIND | CALL (procedure)
//!                    | CALL { subquery } | LOAD CSV
//! writing_clause  ::= CREATE | MERGE | DELETE | DETACH DELETE
//!                    | SET | REMOVE | FOREACH | WITH | RETURN
//!
//! pattern         ::= node_pattern ( rel_pattern node_pattern )*
//! node_pattern    ::= '(' [alias] [':' label]* [properties] ')'
//! rel_pattern     ::= '-[' [alias] [':' type ['|' type]*] [*min..max] ']->'
//!                    | '<-[' ... ']-'   | '-[' ... ']-'
//!
//! expression      ::= or_expr
//! or_expr         ::= xor_expr ( OR xor_expr )*
//! xor_expr        ::= and_expr ( XOR and_expr )*
//! and_expr        ::= not_expr ( AND not_expr )*
//! not_expr        ::= [NOT] comparison
//! comparison      ::= add_sub ( comp_op add_sub )*
//! add_sub         ::= mul_div ( ('+' | '-') mul_div )*
//! mul_div         ::= power ( ('*' | '/' | '%') power )*
//! power           ::= unary ( '^' unary )*
//! unary           ::= ( '+' | '-' )* postfix
//! postfix         ::= primary ( '.' prop | '[' index ']' )*
//! primary         ::= literal | variable | '(' expr ')' | function_call
//!                    | CASE | list_comprehension | pattern_comprehension
//! ```
//!
//! ## Expression Precedence
//!
//! Expressions are parsed with the following precedence (lowest to highest):
//!
//! ```text
//!  Precedence   Operators
//!  ----------   ---------
//!  1 (lowest)   OR
//!  2            XOR
//!  3            AND
//!  4            NOT
//!  5            = <> < > <= >= IN, IS NULL, IS NOT NULL,
//!               STARTS WITH, ENDS WITH, CONTAINS, =~
//!  6            + -
//!  7            * / %
//!  8            ^
//!  9            unary -
//!  10 (highest) property access (.), indexing ([]), function calls
//! ```
//!
//! Expression parsing uses an explicit stack (`Vec<(precedence, Option<tree>)>`)
//! rather than deep call-stack recursion, making it resilient to deeply nested
//! expressions.
//!
//! ## Error Handling
//!
//! Parse errors return `Err(String)` with a descriptive message including
//! the position in the query where the error occurred, the offending token,
//! and context from the original query string.

use crate::entity_type::EntityType;
use crate::identifier_limits::{is_identifier_too_long, validate_identifier_len};
use crate::index::indexer::IndexType;
use crate::parser::ast::{
    AllShortestPaths, ExprIR, QuantifierType, QueryExpr, QueryGraph, QueryIR, QueryNode, QueryPath,
    QueryRelationship, RawQueryIR, ReduceVars, SetItem, ShortestPathInfo,
};
use crate::parser::lexer::{Keyword, Lexer, Token};
use crate::runtime::orderset::OrderSet;
use crate::tree;
use crate::{
    parser::lexer::Token::RParen,
    runtime::{
        functions::{FnType, get_functions},
        value::Value,
    },
};
use itertools::Itertools;
use orx_tree::{DynTree, NodeRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
#[derive(Debug)]
enum ExpressionListType {
    OneOrMore,
    ZeroOrMoreClosedBy(Token),
}

impl ExpressionListType {
    fn is_end_token(
        &self,
        current_token: &Token,
    ) -> bool {
        match self {
            Self::OneOrMore => false,
            Self::ZeroOrMoreClosedBy(token) => token == current_token,
        }
    }
}

#[derive(Clone, Copy)]
struct ParserState {
    pos: usize,
    anon_counter: u32,
}

/// Cypher query parser.
///
/// The parser implements a recursive descent parser with operator precedence
/// for expressions. It consumes tokens from the lexer and builds an AST.
///
/// # Usage
/// ```ignore
/// let mut parser = Parser::new("MATCH (n:Person) RETURN n.name");
/// let ast = parser.parse()?;
/// ```
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    /// Counter for generating unique anonymous variable names
    anon_counter: u32,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given query string.
    #[must_use]
    pub fn new(str: &'a str) -> Self {
        Self {
            lexer: Lexer::new(str),
            anon_counter: 0,
        }
    }

    fn save_state(&self) -> ParserState {
        ParserState {
            pos: self.lexer.pos(true),
            anon_counter: self.anon_counter,
        }
    }

    fn restore_state(
        &mut self,
        state: ParserState,
    ) {
        self.lexer.set_pos(state.pos);
        self.anon_counter = state.anon_counter;
    }

    /// Parses query parameters from CYPHER prefix.
    ///
    /// Handles queries like: `CYPHER param1=value1 param2=value2 MATCH ...`
    /// Returns the parameters map and the remaining query string.
    pub fn parse_parameters(
        &mut self
    ) -> Result<(HashMap<String, DynTree<ExprIR<Arc<String>>>>, &'a str), String> {
        let mut params = HashMap::new();
        while let Ok(Token::IdentifierOrKeyword {
            ident: id,
            keyword: None,
        }) = self.lexer.current()
        {
            if id.as_str() == "CYPHER" {
                self.lexer.next();
                let mut state = self.save_state();
                while let Ok(id) = self.parse_ident() {
                    if !optional_match_token!(self.lexer, Equal) {
                        self.restore_state(state);
                        break;
                    }
                    let value = self.parse_expr(false).map_err(|e| {
                        // An over-long map key inside the value names the exact
                        // thing to fix, so it survives; a syntax error inside a
                        // parameter is more useful reported as the parameter
                        // that failed (see test_params).
                        if is_identifier_too_long(&e) {
                            e
                        } else {
                            format!("Failed to parse the value of parameter '{}'", id.as_str())
                        }
                    })?;
                    params.insert(String::from(id.as_str()), value);
                    state = self.save_state();
                }
            } else {
                break;
            }
        }
        Ok((params, &self.lexer.str[self.lexer.pos(true)..]))
    }

    /// Consumes any trailing semicolons, then verifies end-of-file.
    fn expect_end_of_input(&mut self) -> Result<(), String> {
        while matches!(self.lexer.current()?, Token::Semicolon) {
            self.lexer.next();
        }
        if !matches!(self.lexer.current()?, Token::EndOfFile) {
            return Err("query with more than one statement is not supported".to_string());
        }
        Ok(())
    }

    /// Parses a complete Cypher query.
    ///
    /// This is the main entry point for parsing. It first tries to parse index
    /// operations (CREATE INDEX, DROP INDEX), then falls back to regular queries.
    /// Trailing semicolons are consumed; multi-statement queries (content after
    /// a semicolon) are rejected.
    ///
    /// # Errors
    /// Returns an error string if the query has syntax errors.
    pub fn parse(&mut self) -> Result<RawQueryIR, String> {
        let state = self.save_state();
        let result = if let Some(ir) = self.parse_index_ops()? {
            ir
        } else {
            self.restore_state(state);
            self.parse_query()?
        };

        self.expect_end_of_input()?;

        Ok(result)
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cognitive_complexity)]
    fn parse_index_ops(&mut self) -> Result<Option<RawQueryIR>, String> {
        if optional_match_token!(self.lexer => Create) {
            let fulltext = optional_match_token!(self.lexer => Fulltext);
            let vector = !fulltext && optional_match_token!(self.lexer => Vector);
            let index = optional_match_token!(self.lexer => Index);
            if !index {
                if self.lexer.current_str().eq_ignore_ascii_case("constraint") {
                    return Err(self.lexer.format_error(
                        "Invalid constraint command use the GRAPH.CONSTRAINT command instead",
                    ));
                }
                return Ok(None);
            }
            if !fulltext && !vector && optional_match_token!(self.lexer => On) {
                match_token!(self.lexer, Colon);
                let label = self.parse_ident()?;
                match_token!(self.lexer, LParen);
                let mut attrs = vec![self.parse_ident()?];
                while optional_match_token!(self.lexer, Comma) {
                    attrs.push(self.parse_ident()?);
                }
                match_token!(self.lexer, RParen);
                let index_type = IndexType::Range;
                let entity_type = EntityType::Node;
                return Ok(Some(QueryIR::CreateIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type,
                    options: None,
                }));
            }
            match_token!(self.lexer => For);
            match_token!(self.lexer, LParen);
            let (nkey, label, entity_type) = if optional_match_token!(self.lexer, RParen) {
                match_token!(self.lexer, Dash);
                match_token!(self.lexer, LBrace);
                let nkey = self.parse_ident()?;
                match_token!(self.lexer, Colon);
                let label = self.parse_ident()?;
                match_token!(self.lexer, RBrace);
                match_token!(self.lexer, Dash);
                optional_match_token!(self.lexer, GreaterThan);
                match_token!(self.lexer, LParen);
                match_token!(self.lexer, RParen);
                (nkey, label, EntityType::Relationship)
            } else {
                let nkey = self.parse_ident()?;
                if !matches!(self.lexer.current()?, Token::Colon) {
                    return Err(self.lexer.format_error(&format!(
                        "Invalid input '{}': expected a label",
                        self.lexer.current_str()
                    )));
                }
                self.lexer.next();
                let label = self.parse_ident()?;
                match_token!(self.lexer, RParen);
                (nkey, label, EntityType::Node)
            };
            match_token!(self.lexer => On);
            match_token!(self.lexer, LParen);
            let key = self.parse_ident()?;
            self.match_dot_property_separator()?;
            if nkey.as_str() != key.as_str() {
                return Err(self.lexer.format_error(&format!("'{key}' not defined")));
            }
            let mut attrs = vec![self.parse_property_name()?];
            while optional_match_token!(self.lexer, Comma) {
                let key = self.parse_ident()?;
                self.match_dot_property_separator()?;
                if nkey.as_str() != key.as_str() {
                    return Err(self.lexer.format_error(&format!("'{key}' not defined")));
                }
                attrs.push(self.parse_property_name()?);
            }
            match_token!(self.lexer, RParen);
            let index_type = if fulltext {
                IndexType::Fulltext
            } else if vector {
                IndexType::Vector
            } else {
                IndexType::Range
            };
            let options = if (vector || fulltext) && optional_match_token!(self.lexer => Options) {
                Some(Arc::new(self.parse_map()?))
            } else {
                None
            };
            return Ok(Some(QueryIR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            }));
        }
        if optional_match_token!(self.lexer => Drop) {
            let fulltext = optional_match_token!(self.lexer => Fulltext);
            let vector = !fulltext && optional_match_token!(self.lexer => Vector);
            let index = optional_match_token!(self.lexer => Index);
            if !index {
                if self.lexer.current_str().eq_ignore_ascii_case("constraint") {
                    return Err(self.lexer.format_error(
                        "Invalid constraint command use the GRAPH.CONSTRAINT command instead",
                    ));
                }
                return Ok(None);
            }
            if !fulltext && !vector && optional_match_token!(self.lexer => On) {
                match_token!(self.lexer, Colon);
                let label = self.parse_ident()?;
                match_token!(self.lexer, LParen);
                let mut attrs = vec![self.parse_ident()?];
                while optional_match_token!(self.lexer, Comma) {
                    attrs.push(self.parse_ident()?);
                }
                match_token!(self.lexer, RParen);
                let index_type = IndexType::Range;
                let entity_type = EntityType::Node;
                return Ok(Some(QueryIR::DropIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type,
                }));
            }
            match_token!(self.lexer => For);
            match_token!(self.lexer, LParen);
            let (nkey, label, entity_type) = if optional_match_token!(self.lexer, RParen) {
                match_token!(self.lexer, Dash);
                match_token!(self.lexer, LBrace);
                let nkey = self.parse_ident()?;
                match_token!(self.lexer, Colon);
                let label = self.parse_ident()?;
                match_token!(self.lexer, RBrace);
                match_token!(self.lexer, Dash);
                optional_match_token!(self.lexer, GreaterThan);
                match_token!(self.lexer, LParen);
                match_token!(self.lexer, RParen);
                (nkey, label, EntityType::Relationship)
            } else {
                let nkey = self.parse_ident()?;
                if !matches!(self.lexer.current()?, Token::Colon) {
                    return Err(self.lexer.format_error(&format!(
                        "Invalid input '{}': expected a label",
                        self.lexer.current_str()
                    )));
                }
                self.lexer.next();
                let label = self.parse_ident()?;
                match_token!(self.lexer, RParen);
                (nkey, label, EntityType::Node)
            };
            match_token!(self.lexer => On);
            match_token!(self.lexer, LParen);
            let key = self.parse_ident()?;
            self.match_dot_property_separator()?;
            if nkey.as_str() != key.as_str() {
                return Err(self.lexer.format_error(&format!("'{key}' not defined")));
            }
            let mut attrs = vec![self.parse_property_name()?];
            while optional_match_token!(self.lexer, Comma) {
                let key = self.parse_ident()?;
                self.match_dot_property_separator()?;
                if nkey.as_str() != key.as_str() {
                    return Err(self.lexer.format_error(&format!("'{key}' not defined")));
                }
                attrs.push(self.parse_property_name()?);
            }
            match_token!(self.lexer, RParen);
            let index_type = if fulltext {
                IndexType::Fulltext
            } else if vector {
                IndexType::Vector
            } else {
                IndexType::Range
            };
            return Ok(Some(QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            }));
        }
        Ok(None)
    }

    /// Parses a complete query, handling UNION if present.
    ///
    /// A Cypher query may consist of multiple sub-queries joined by UNION:
    ///
    /// ```cypher
    /// MATCH (n:Person) RETURN n.name AS name
    /// UNION
    /// MATCH (n:Company) RETURN n.title AS name
    /// ```
    ///
    /// Each sub-query is parsed by `parse_single_query()`.  If UNION appears
    /// after the first sub-query's RETURN, additional sub-queries are parsed
    /// and wrapped in `QueryIR::Union`.
    fn parse_query(&mut self) -> Result<RawQueryIR, String> {
        let first = self.parse_single_query()?;
        if optional_match_token!(self.lexer => Union) {
            let all = optional_match_token!(self.lexer => All);
            let mut branches = vec![first];
            branches.push(self.parse_single_query()?);
            while optional_match_token!(self.lexer => Union) {
                let next_all = optional_match_token!(self.lexer => All);
                if next_all != all {
                    return Err(self
                        .lexer
                        .format_error("Invalid combination of UNION and UNION ALL."));
                }
                branches.push(self.parse_single_query()?);
            }
            return Ok(QueryIR::Union { branches, all });
        }
        Ok(first)
    }

    /// Parses a single sub-query (clauses up through an optional RETURN).
    ///
    /// This handles the reading/writing clause loops and WITH/RETURN
    /// projection.  Called once for standalone queries and once per branch
    /// in a UNION query.
    fn parse_single_query(&mut self) -> Result<RawQueryIR, String> {
        let mut clauses = Vec::new();
        let mut write = false;
        loop {
            while let Token::IdentifierOrKeyword {
                keyword:
                    Some(
                        Keyword::Optional
                        | Keyword::Match
                        | Keyword::Unwind
                        | Keyword::Call
                        | Keyword::Load,
                    ),
                ..
            } = self.lexer.current()?
            {
                if matches!(
                    self.lexer.current()?,
                    Token::IdentifierOrKeyword {
                        keyword: Some(Keyword::Call),
                        ..
                    }
                ) {
                    self.lexer.next();
                    clauses.extend(self.parse_call_clause()?);
                } else {
                    let clause = self.parse_reading_clasue()?;
                    clauses.push(clause);
                }
            }
            while let Token::IdentifierOrKeyword {
                keyword:
                    Some(
                        Keyword::Create
                        | Keyword::Merge
                        | Keyword::Delete
                        | Keyword::Detach
                        | Keyword::Set
                        | Keyword::Remove
                        | Keyword::Foreach,
                    ),
                ..
            } = self.lexer.current()?
            {
                write = true;
                let clause = self.parse_writing_clause()?;
                clauses.push(clause);
            }
            if optional_match_token!(self.lexer => With) {
                clauses.push(self.parse_with_clause(write)?);
            } else {
                // After updating clauses, a reading clause requires WITH.
                if write {
                    match self.lexer.current()? {
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Match | Keyword::Optional),
                            ..
                        } => {
                            return Err(self.lexer.format_error(
                                "A WITH clause is required to introduce MATCH after an updating clause.",
                            ));
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Unwind),
                            ..
                        } => {
                            return Err(self.lexer.format_error(
                                "A WITH clause is required to introduce UNWIND after an updating clause.",
                            ));
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Call),
                            ..
                        } => {
                            // Peek ahead to distinguish CALL { subquery } from CALL procedure()
                            let state = self.save_state();
                            self.lexer.next(); // consume CALL
                            let is_subquery = matches!(self.lexer.current(), Ok(Token::LBracket));
                            self.restore_state(state);
                            let msg = if is_subquery {
                                "A WITH clause is required to introduce CALL SUBQUERY after an updating clause."
                            } else {
                                "A WITH clause is required to introduce CALL after an updating clause."
                            };
                            return Err(self.lexer.format_error(msg));
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Load),
                            ..
                        } => {
                            return Err(self.lexer.format_error(
                                "A WITH clause is required to introduce LOAD CSV after an updating clause.",
                            ));
                        }
                        _ => {}
                    }
                }
                break;
            }
            write = false;
        }
        if optional_match_token!(self.lexer => Return) {
            clauses.push(self.parse_return_clause(write)?);
            write = false;
            // After RETURN, only UNION, semicolons, end-of-file, or closing brace may follow.
            match self.lexer.current()? {
                Token::EndOfFile
                | Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Union),
                    ..
                }
                | Token::Semicolon
                | Token::RBracket => {}
                _ => {
                    return Err(self
                        .lexer
                        .format_error("Unexpected clause following RETURN"));
                }
            }
        }
        if !matches!(
            self.lexer.current()?,
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Union),
                ..
            } | Token::Semicolon
                | Token::RBracket
        ) {
            match_token!(self.lexer, EndOfFile);
        }
        Ok(QueryIR::Query { clauses, write })
    }

    fn parse_reading_clasue(&mut self) -> Result<RawQueryIR, String> {
        if optional_match_token!(self.lexer => Optional) {
            match_token!(self.lexer => Match);
            return self.parse_match_clause(true);
        }
        match self.lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Match),
                ..
            } => {
                self.lexer.next();
                optional_match_token!(self.lexer => Match);
                self.parse_match_clause(false)
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Unwind),
                ..
            } => {
                self.lexer.next();
                self.parse_unwind_clause()
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Load),
                ..
            } => {
                self.lexer.next();
                match_token!(self.lexer => Csv);
                let headers = optional_match_token!(self.lexer => With)
                    && optional_match_token!(self.lexer => Headers);
                match_token!(self.lexer => From);
                let file_path = Arc::new(self.parse_expr(false)?);
                match_token!(self.lexer => As);
                let ident = self.parse_ident()?;
                // Support standard Cypher FIELDTERMINATOR after AS
                let delimiter = if optional_match_token!(self.lexer => Fieldterminator) {
                    Arc::new(self.parse_expr(false)?)
                } else {
                    Arc::new(tree!(ExprIR::Constant(Value::String(Arc::new(
                        String::from(',')
                    )))))
                };
                Ok(QueryIR::LoadCsv {
                    file_path,
                    headers,
                    delimiter,
                    var: ident,
                })
            }
            _ => unreachable!(),
        }
    }

    fn parse_writing_clause(&mut self) -> Result<RawQueryIR, String> {
        match self.lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Create),
                ..
            } => {
                self.lexer.next();
                self.parse_create_clause()
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Merge),
                ..
            } => {
                self.lexer.next();
                self.parse_merge_clause()
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Detach | Keyword::Delete),
                ..
            } => {
                let is_detach = optional_match_token!(self.lexer => Detach);
                match_token!(self.lexer => Delete);
                self.parse_delete_clause(is_detach)
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Set),
                ..
            } => {
                self.lexer.next();
                self.parse_set_clause()
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Remove),
                ..
            } => {
                self.lexer.next();
                self.parse_remove_clause()
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Foreach),
                ..
            } => {
                self.lexer.next();
                self.parse_foreach_clause()
            }
            _ => unreachable!(),
        }
    }

    fn parse_call_clause(&mut self) -> Result<Vec<RawQueryIR>, String> {
        // CALL { subquery } — parse body as a self-contained query
        if self.lexer.current()? == Token::LBracket {
            self.lexer.next();
            let body = self.parse_query()?;
            match_token!(self.lexer, RBracket);
            let is_returning = Self::body_has_return(&body);
            return Ok(vec![QueryIR::CallSubquery {
                body: Box::new(body),
                is_returning,
                remap: vec![],
            }]);
        }

        // CALL procedure() — existing procedure call parsing
        // Parentheses are optional: `CALL db.labels` is equivalent to `CALL db.labels()`.
        let function_name = self.parse_dotted_ident()?;
        validate_identifier_len(function_name.as_str(), "Procedure name")?;
        let func = get_functions().get(function_name.as_str(), &FnType::Procedure(vec![]))?;
        let args: Vec<Arc<_>> = if optional_match_token!(self.lexer, LParen) {
            self.parse_expression_list(ExpressionListType::ZeroOrMoreClosedBy(RParen), false)?
                .into_iter()
                .map(Arc::new)
                .collect()
        } else {
            vec![]
        };
        func.validate(args.len())
            .map_err(|e| e.replace("function", "procedure"))?;
        let mut named_outputs = vec![];
        let mut yield_aliases: Vec<Option<Arc<String>>> = vec![];
        let (filter, yielded) = if optional_match_token!(self.lexer => Yield) {
            let ident = self.parse_ident()?;
            if optional_match_token!(self.lexer => As) {
                let alias = self.parse_ident()?;
                yield_aliases.push(Some(ident));
                named_outputs.push(alias);
            } else {
                yield_aliases.push(None);
                named_outputs.push(ident);
            }
            while optional_match_token!(self.lexer, Comma) {
                let ident = self.parse_ident()?;
                if optional_match_token!(self.lexer => As) {
                    let alias = self.parse_ident()?;
                    yield_aliases.push(Some(ident));
                    named_outputs.push(alias);
                } else {
                    yield_aliases.push(None);
                    named_outputs.push(ident);
                }
            }
            (self.parse_where()?, true)
        } else if let FnType::Procedure(defult_outputs) = &func.fn_type {
            for output in defult_outputs {
                named_outputs.push(Arc::new(output.clone()));
                yield_aliases.push(None);
            }
            (None, false)
        } else {
            (None, false)
        };

        Ok(vec![QueryIR::Call {
            func,
            args,
            yields: named_outputs,
            yield_aliases,
            filter,
            explicit_yield: yielded,
        }])
    }

    /// Check if a parsed query body ends with a RETURN clause.
    fn body_has_return(ir: &RawQueryIR) -> bool {
        match ir {
            QueryIR::Query { clauses, .. } => clauses
                .last()
                .is_some_and(|c| matches!(c, QueryIR::Return { .. })),
            QueryIR::Union { branches, .. } => branches.iter().all(Self::body_has_return),
            _ => false,
        }
    }

    fn parse_dotted_ident(&mut self) -> Result<Arc<String>, String> {
        let mut idents = vec![self.parse_ident()?];
        while self.lexer.current()? == Token::Dot {
            self.lexer.next();
            idents.push(self.parse_ident()?);
        }
        Ok(Arc::new(
            idents.iter().map(|label| label.as_str()).join("."),
        ))
    }

    fn parse_match_clause(
        &mut self,
        optional: bool,
    ) -> Result<RawQueryIR, String> {
        Ok(QueryIR::Match {
            pattern: self.parse_pattern(&Keyword::Match)?,
            filter: self.parse_where()?,
            optional,
        })
    }

    fn parse_unwind_clause(&mut self) -> Result<RawQueryIR, String> {
        let list = Arc::new(self.parse_expr(false)?);
        match_token!(self.lexer => As);
        let ident = self.parse_ident()?;
        Ok(QueryIR::Unwind {
            expr: list,
            var: ident,
        })
    }

    fn parse_create_clause(&mut self) -> Result<RawQueryIR, String> {
        Ok(QueryIR::Create(self.parse_pattern(&Keyword::Create)?))
    }

    fn parse_merge_clause(&mut self) -> Result<RawQueryIR, String> {
        let pattern = self.parse_pattern(&Keyword::Merge)?;
        let mut on_match_set_items = vec![];
        let mut on_create_set_items = vec![];
        while optional_match_token!(self.lexer => On) {
            if optional_match_token!(self.lexer => Match) {
                match_token!(self.lexer => Set);
                self.parse_set_items(&mut on_match_set_items)?;
            } else if optional_match_token!(self.lexer => Create) {
                match_token!(self.lexer => Set);
                self.parse_set_items(&mut on_create_set_items)?;
            } else {
                return Err(self.lexer.format_error("Expected MATCH or CREATE after ON"));
            }
        }
        Ok(QueryIR::Merge {
            pattern,
            on_create: on_create_set_items,
            on_match: on_match_set_items,
        })
    }

    fn parse_delete_clause(
        &mut self,
        is_detach: bool,
    ) -> Result<RawQueryIR, String> {
        let mut exprs = self
            .parse_expression_list(ExpressionListType::OneOrMore, false)?
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        let mut any_detach = is_detach;

        // Combine consecutive DELETE clauses into one
        while matches!(
            self.lexer.current()?,
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Delete | Keyword::Detach),
                ..
            }
        ) {
            let next_is_detach = optional_match_token!(self.lexer => Detach);
            match_token!(self.lexer => Delete);
            let more_exprs = self
                .parse_expression_list(ExpressionListType::OneOrMore, false)?
                .into_iter()
                .map(Arc::new)
                .collect::<Vec<_>>();
            exprs.extend(more_exprs);
            any_detach = any_detach || next_is_detach;
        }

        Ok(QueryIR::Delete {
            exprs,
            detach: any_detach,
        })
    }

    fn parse_where(&mut self) -> Result<Option<QueryExpr<Arc<String>>>, String> {
        if let Token::IdentifierOrKeyword {
            keyword: Some(Keyword::Where),
            ..
        } = self.lexer.current()?
        {
            self.lexer.next();
            let expr = self.parse_expr(true)?;
            if let Some(func) = Self::find_aggregate_name(&expr) {
                return Err(self
                    .lexer
                    .format_error(&format!("Invalid use of aggregating function '{func}'")));
            }
            return Ok(Some(Arc::new(expr)));
        }
        Ok(None)
    }

    /// Parse ORDER BY, SKIP, and LIMIT clauses shared by WITH and RETURN.
    fn parse_orderby_skip_limit(
        &mut self
    ) -> Result<
        (
            Vec<(QueryExpr<Arc<String>>, bool)>,
            Option<QueryExpr<Arc<String>>>,
            Option<QueryExpr<Arc<String>>>,
        ),
        String,
    > {
        let orderby = if optional_match_token!(self.lexer => Order) {
            self.parse_orderby()?
        } else {
            vec![]
        };
        let skip = if optional_match_token!(self.lexer => Skip) {
            let skip = Arc::new(self.parse_expr(false)?);
            match skip.root().data() {
                ExprIR::Constant(Value::Int(i)) => {
                    if *i < 0 {
                        return Err(self.lexer.format_error(
                            "SKIP specified value of invalid type, must be a positive integer",
                        ));
                    }
                }
                ExprIR::Parameter(_) => {}
                _ => {
                    return Err(self.lexer.format_error(
                        "SKIP specified value of invalid type, must be a positive integer",
                    ));
                }
            }
            Some(skip)
        } else {
            None
        };
        let limit = if optional_match_token!(self.lexer => Limit) {
            let limit = Arc::new(self.parse_expr(false)?);
            match limit.root().data() {
                ExprIR::Constant(Value::Int(i)) => {
                    if *i < 0 {
                        return Err(self.lexer.format_error(
                            "LIMIT specified value of invalid type, must be a positive integer",
                        ));
                    }
                }
                ExprIR::Parameter(_) => {}
                _ => {
                    return Err(self.lexer.format_error(
                        "LIMIT specified value of invalid type, must be a positive integer",
                    ));
                }
            }
            Some(limit)
        } else {
            None
        };
        Ok((orderby, skip, limit))
    }

    fn parse_with_clause(
        &mut self,
        write: bool,
    ) -> Result<RawQueryIR, String> {
        let distinct = optional_match_token!(self.lexer => Distinct);
        let (all, exprs) = if optional_match_token!(self.lexer, Star) {
            // WITH * carries forward the current named_in_scope unchanged
            if optional_match_token!(self.lexer, Comma) {
                (true, self.parse_named_exprs(true)?)
            } else {
                (true, vec![])
            }
        } else {
            (false, self.parse_named_exprs(true)?)
        };
        let (orderby, skip, limit) = self.parse_orderby_skip_limit()?;
        let filter = self.parse_where()?;
        Ok(QueryIR::With {
            distinct,
            all,
            exprs,
            copy_from_parent: Vec::new(),
            orderby,
            skip,
            limit,
            filter,
            write,
        })
    }

    fn parse_return_clause(
        &mut self,
        write: bool,
    ) -> Result<RawQueryIR, String> {
        let distinct = optional_match_token!(self.lexer => Distinct);
        let (all, exprs) = if optional_match_token!(self.lexer, Star) {
            if optional_match_token!(self.lexer, Comma) {
                (true, self.parse_named_exprs(false)?)
            } else {
                (true, vec![])
            }
        } else {
            (false, self.parse_named_exprs(false)?)
        };
        let (orderby, skip, limit) = self.parse_orderby_skip_limit()?;

        Ok(QueryIR::Return {
            distinct,
            all,
            exprs,
            copy_from_parent: Vec::new(),
            orderby,
            skip,
            limit,
            write,
        })
    }

    /// Adds a pattern node to `query_graph`, folding a repeated alias into the
    /// entry that is already there.
    ///
    /// In MATCH every occurrence of an alias contributes predicates, so labels
    /// and inline attributes accumulate: `MATCH (n:A) MATCH (n:B)` requires
    /// both labels, `MATCH (n {v: 1}), (n {w: 2})` both properties. CREATE and
    /// MERGE keep the first occurrence — a repeated alias there refers back to
    /// the entity being created, it does not add predicates.
    fn add_pattern_node(
        query_graph: &mut QueryGraph<Arc<String>, Arc<String>, Arc<String>>,
        nodes_alias: &mut HashSet<Arc<String>>,
        node: &Arc<QueryNode<Arc<String>, Arc<String>>>,
        clause: &Keyword,
    ) {
        if nodes_alias.insert(node.alias.clone()) {
            query_graph.add_node(node.clone());
        } else if *clause == Keyword::Match {
            query_graph.merge_node(node);
        }
    }

    fn parse_pattern(
        &mut self,
        clause: &Keyword,
    ) -> Result<QueryGraph<Arc<String>, Arc<String>, Arc<String>>, String> {
        let mut query_graph = QueryGraph::default();
        let mut nodes_alias = HashSet::new();
        loop {
            if let Ok(ident) = self.parse_ident() {
                match_token!(self.lexer, Equal);

                // Check for shortestPath/allShortestPaths in MATCH clause
                if let Token::IdentifierOrKeyword { ident: value, .. } = self.lexer.current()? {
                    if value.eq_ignore_ascii_case("shortestPath") {
                        return Err(self.lexer.format_error(
                            "FalkorDB currently only supports shortestPaths in WITH or RETURN clauses",
                        ));
                    }
                    if value.eq_ignore_ascii_case("allShortestPaths") {
                        // Parse allShortestPaths((src)-[*]->(dst))
                        self.lexer.next(); // consume 'allShortestPaths'
                        match_token!(self.lexer, LParen); // consume outer '('

                        let mut vars = vec![];
                        let left = self.parse_node_pattern()?;
                        let left_alias = left.alias.clone();
                        vars.push(left.alias.clone());
                        Self::add_pattern_node(&mut query_graph, &mut nodes_alias, &left, clause);

                        // Parse relationship chain inside allShortestPaths.
                        // Multiple relationships are allowed: fixed-length prefix
                        // relationships are added as regular relationships, and the
                        // variable-length one gets the allShortestPaths flag.
                        if !matches!(self.lexer.current()?, Token::Dash | Token::LessThan) {
                            return Err(self
                                .lexer
                                .format_error("allShortestPaths requires a relationship pattern"));
                        }

                        let mut prev_node = left;
                        let mut asp_found = false;
                        loop {
                            if !matches!(self.lexer.current()?, Token::Dash | Token::LessThan) {
                                break;
                            }
                            let (relationship, right) =
                                self.parse_relationship_pattern(prev_node, clause)?;

                            let is_var_len =
                                relationship.min_hops.is_some() || relationship.max_hops.is_some();

                            if is_var_len {
                                // This is the allShortestPaths relationship
                                if asp_found {
                                    return Err(self.lexer.format_error(
                                        "allShortestPaths supports at most one variable-length relationship",
                                    ));
                                }
                                asp_found = true;

                                // Validate min_hops
                                if let Some(min) = relationship.min_hops
                                    && min > 1
                                {
                                    return Err(self.lexer.format_error(
                                        "allShortestPaths(...) does not support a minimal length different from 1",
                                    ));
                                }

                                let mut new_rel = QueryRelationship::new(
                                    relationship.alias.clone(),
                                    relationship.types.clone(),
                                    relationship.attrs.clone(),
                                    relationship.from.clone(),
                                    relationship.to.clone(),
                                    relationship.bidirectional,
                                    Some(1),
                                    relationship.max_hops,
                                );
                                new_rel.all_shortest_paths = if !relationship.bidirectional
                                    && relationship.from.alias != left_alias
                                {
                                    AllShortestPaths::Reversed
                                } else {
                                    AllShortestPaths::Forward
                                };
                                let relationship = Arc::new(new_rel);

                                vars.push(relationship.alias.clone());
                                vars.push(right.alias.clone());
                                if !query_graph.add_relationship(relationship.clone())
                                    && clause == &Keyword::Match
                                {
                                    return Err(format!(
                                        "Cannot use the same relationship variable '{}' for multiple patterns.",
                                        relationship.alias.as_str()
                                    ));
                                }
                            } else {
                                // Fixed-length prefix/suffix relationship
                                vars.push(relationship.alias.clone());
                                vars.push(right.alias.clone());
                                if !query_graph.add_relationship(relationship.clone())
                                    && clause == &Keyword::Match
                                {
                                    return Err(format!(
                                        "Cannot use the same relationship variable '{}' for multiple patterns.",
                                        relationship.alias.as_str()
                                    ));
                                }
                            }
                            Self::add_pattern_node(
                                &mut query_graph,
                                &mut nodes_alias,
                                &right,
                                clause,
                            );
                            prev_node = right;
                        }

                        if !asp_found {
                            return Err(self.lexer.format_error(
                                "allShortestPaths requires a variable-length relationship pattern",
                            ));
                        }

                        match_token!(self.lexer, RParen); // consume outer ')'
                        query_graph.add_path(Arc::new(QueryPath::new(ident, vars)));

                        // Continue to next pattern or end
                        if self.lexer.current()? != Token::Comma {
                            break;
                        }
                        self.lexer.next(); // consume comma
                        continue;
                    }
                }

                let mut vars = vec![];
                let mut left = self.parse_node_pattern()?;
                vars.push(left.alias.clone());
                Self::add_pattern_node(&mut query_graph, &mut nodes_alias, &left, clause);
                loop {
                    if let Token::Dash | Token::LessThan = self.lexer.current()? {
                        let (relationship, right) =
                            self.parse_relationship_pattern(left, clause)?;
                        vars.push(relationship.alias.clone());
                        vars.push(right.alias.clone());
                        left = right.clone();
                        if !query_graph.add_relationship(relationship.clone())
                            && clause == &Keyword::Match
                        {
                            return Err(format!(
                                "Cannot use the same relationship variable '{}' for multiple patterns.",
                                relationship.alias.as_str()
                            ));
                        }
                        Self::add_pattern_node(&mut query_graph, &mut nodes_alias, &right, clause);
                    } else {
                        query_graph.add_path(Arc::new(QueryPath::new(ident, vars)));
                        break;
                    }
                }
            } else {
                let mut left = self.parse_node_pattern()?;

                Self::add_pattern_node(&mut query_graph, &mut nodes_alias, &left, clause);
                while let Token::Dash | Token::LessThan = self.lexer.current()? {
                    let (relationship, right) = self.parse_relationship_pattern(left, clause)?;
                    left = right.clone();
                    if !query_graph.add_relationship(relationship.clone()) {
                        if clause == &Keyword::Match {
                            return Err(format!(
                                "Cannot use the same relationship variable '{}' for multiple patterns.",
                                relationship.alias.as_str()
                            ));
                        }
                        if clause == &Keyword::Create {
                            return Err(format!(
                                "The bound variable '{}' can't be redeclared in a CREATE clause",
                                relationship.alias.as_str()
                            ));
                        }
                    }
                    Self::add_pattern_node(&mut query_graph, &mut nodes_alias, &right, clause);
                }
            }

            if *clause == Keyword::Merge {
                break;
            }

            match self.lexer.current()? {
                Token::Comma => {
                    self.lexer.next();
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(token),
                    ..
                } => {
                    if token == *clause {
                        self.lexer.next();
                        continue;
                    }
                    break;
                }
                _ => break,
            }
        }

        Ok(query_graph)
    }

    fn parse_case_expression(&mut self) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        self.lexer.next();
        let mut children = vec![];
        let has_subject = !matches!(
            self.lexer.current()?,
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::When),
                ..
            }
        );
        if has_subject {
            children.push(self.parse_expr(false)?);
        }
        let mut conditions = vec![];
        while optional_match_token!(self.lexer => When) {
            conditions.push(self.parse_expr(false)?);
            match_token!(self.lexer => Then);
            conditions.push(self.parse_expr(false)?);
        }
        if conditions.is_empty() {
            return Err(self.lexer.format_error("Invalid input"));
        }
        children.push(tree!(ExprIR::List ; conditions));
        if optional_match_token!(self.lexer => Else) {
            children.push(self.parse_expr(false)?);
        } else {
            children.push(tree!(ExprIR::Constant(Value::Null)));
        }
        match_token!(self.lexer => End);
        Ok(tree!(ExprIR::Case { has_subject }; children))
    }

    fn parse_quantifier_expr(
        &mut self,
        allow_pattern_predicate: bool,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let quantifier_type = match self.lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::All),
                ..
            } => {
                self.lexer.next();
                QuantifierType::All
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Any),
                ..
            } => {
                self.lexer.next();
                QuantifierType::Any
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::None),
                ..
            } => {
                self.lexer.next();
                QuantifierType::None
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Single),
                ..
            } => {
                self.lexer.next();
                QuantifierType::Single
            }
            _ => unreachable!(),
        };

        match_token!(self.lexer, LParen);
        let var = self.parse_ident()?;
        match_token!(self.lexer => In);
        let expr = self.parse_expr(allow_pattern_predicate)?;
        if !optional_match_token!(self.lexer => Where) {
            return Err(self.lexer.format_error(&format!(
                "'{}' function requires a WHERE predicate",
                quantifier_type.to_string().to_uppercase()
            )));
        }
        let condition = self.parse_expr(allow_pattern_predicate)?;
        match_token!(self.lexer, RParen);
        Ok(tree!(
            ExprIR::Quantifier {
                quantifier_type,
                var
            },
            expr,
            condition
        ))
    }

    /// Parses `reduce(acc = init, var IN list | expr)` after the opening `(`.
    fn parse_reduce_expr(
        &mut self,
        allow_pattern_predicate: bool,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        // accumulator name
        let Ok(acc_var) = self.parse_ident() else {
            return Err(self.lexer.format_error("Unknown function 'reduce'"));
        };

        // '=' for accumulator init
        if !optional_match_token!(self.lexer, Equal) {
            return Err(self.lexer.format_error("Unknown function 'reduce'"));
        }

        // init expression
        let init_expr = self.parse_expr(allow_pattern_predicate)?;

        // Check for aggregate functions in the init expression
        if let Some(func) = Self::find_aggregate_name(&init_expr) {
            return Err(self
                .lexer
                .format_error(&format!("Invalid use of aggregating function '{func}'")));
        }

        // ','
        if !optional_match_token!(self.lexer, Comma) {
            return Err(self.lexer.format_error("Unknown function 'reduce'"));
        }

        // iteration variable
        let iter_var = self.parse_ident()?;

        // IN
        if !optional_match_token!(self.lexer => In) {
            return Err(self.lexer.format_error("Unknown function 'reduce'"));
        }

        // list expression
        let list_expr = self.parse_expr(allow_pattern_predicate)?;

        // Check for aggregate functions in the list expression
        if let Some(func) = Self::find_aggregate_name(&list_expr) {
            return Err(self
                .lexer
                .format_error(&format!("Invalid use of aggregating function '{func}'")));
        }

        // '|'
        if !optional_match_token!(self.lexer, Pipe) {
            return Err(self.lexer.format_error("Unknown function 'reduce'"));
        }

        // body expression
        let body_expr = self.parse_expr(allow_pattern_predicate)?;

        // Check for aggregate functions in the body expression
        if let Some(func) = Self::find_aggregate_name(&body_expr) {
            return Err(self
                .lexer
                .format_error(&format!("Invalid use of aggregating function '{func}'")));
        }

        // ')'
        match_token!(self.lexer, RParen);

        Ok(tree!(
            ExprIR::Reduce(Box::new(ReduceVars {
                accumulator: acc_var,
                iterator: iter_var
            })),
            init_expr,
            list_expr,
            body_expr
        ))
    }

    /// Parses `shortestPath((src)-[rel:TYPE*min..max]->(dst))` after the
    /// opening `(`.
    fn parse_shortest_path_expr(&mut self) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let fn_name = "shortestPath";

        // Verify pattern starts with `(` — otherwise it's not a valid shortestPath call
        if self.lexer.current()? != Token::LParen {
            return Err(self
                .lexer
                .format_error(&format!("Unknown function '{fn_name}'")));
        }

        // Parse source node: (ident)
        match_token!(self.lexer, LParen);
        let Ok(src_var) = self.parse_ident() else {
            return Err(self
                .lexer
                .format_error(&format!("A {fn_name} requires bound nodes")));
        };
        // Check for inline node properties (not allowed)
        if self.lexer.current()? == Token::LBracket {
            return Err(self
                .lexer
                .format_error(&format!("A {fn_name} requires bound nodes")));
        }
        match_token!(self.lexer, RParen);

        // Parse direction and relationship: -[...]->, <-[...]-
        let is_incoming = optional_match_token!(self.lexer, LessThan);
        match_token!(self.lexer, Dash);

        // Parse relationship details in [...]
        let has_details = optional_match_token!(self.lexer, LBrace);
        let (rel_types, min_hops, max_hops, has_edge_filter) = if has_details {
            // Optional alias (ignored)
            let _alias = self.parse_ident().ok();

            // Relationship types
            let mut types = Vec::new();
            if optional_match_token!(self.lexer, Colon) {
                loop {
                    types.push(self.parse_ident()?);
                    let pipe = optional_match_token!(self.lexer, Pipe);
                    let colon = optional_match_token!(self.lexer, Colon);
                    if pipe || colon {
                        continue;
                    }
                    break;
                }
            }

            // Variable length * and range
            let (min, max) = if optional_match_token!(self.lexer, Star) {
                let start = if let Token::Integer(i) = self.lexer.current()? {
                    self.lexer.next();
                    Some(i as u32)
                } else {
                    None
                };
                if optional_match_token!(self.lexer, DotDot) {
                    let end = if let Token::Integer(i) = self.lexer.current()? {
                        self.lexer.next();
                        Some(i as u32)
                    } else {
                        None
                    };
                    (start.unwrap_or(1), end)
                } else if let Some(exact) = start {
                    (exact, Some(exact))
                } else {
                    (1, None) // [*] = 1..infinity
                }
            } else {
                (1, Some(1)) // no *, fixed 1-hop
            };

            // Check for edge filter properties (not allowed)
            let has_filter = if let Token::Parameter(_) = self.lexer.current()? {
                true
            } else {
                self.lexer.current()? == Token::LBracket
            };

            // Skip filter content if present (consume until we reach `]`)
            if has_filter {
                let mut depth = 0i32;
                loop {
                    let tok = self.lexer.current()?;
                    match tok {
                        Token::LBracket => {
                            // Nested `{`
                            depth += 1;
                            self.lexer.next();
                        }
                        Token::RBracket => {
                            // Closing `}`
                            depth -= 1;
                            self.lexer.next();
                            if depth <= 0 {
                                break;
                            }
                        }
                        Token::RBrace => {
                            // `]` - stop before consuming it
                            break;
                        }
                        _ => {
                            self.lexer.next();
                        }
                    }
                }
            }

            match_token!(self.lexer, RBrace);
            (types, min, max, has_filter)
        } else {
            // -[]-  bare relationship
            (vec![], 1, Some(1), false)
        };

        match_token!(self.lexer, Dash);
        let is_outgoing = optional_match_token!(self.lexer, GreaterThan);

        // Determine direction
        let directed = is_incoming || is_outgoing;

        // Parse destination node: (ident)
        match_token!(self.lexer, LParen);
        let Ok(dst_var) = self.parse_ident() else {
            // Could be empty `()` in a multi-relationship pattern like (a)-[]->()(b)
            // or simply an anonymous node
            if self.lexer.current()? == Token::RParen {
                // Empty node: check if there's another relationship after it
                self.lexer.next(); // consume )
                if matches!(self.lexer.current()?, Token::Dash | Token::LessThan) {
                    return Err(self.lexer.format_error(&format!(
                        "{fn_name} requires a path containing a single relationship"
                    )));
                }
                // It's just () as dest
                return Err(self
                    .lexer
                    .format_error(&format!("A {fn_name} requires bound nodes")));
            }
            return Err(self
                .lexer
                .format_error(&format!("A {fn_name} requires bound nodes")));
        };
        if self.lexer.current()? == Token::LBracket {
            return Err(self
                .lexer
                .format_error(&format!("A {fn_name} requires bound nodes")));
        }
        match_token!(self.lexer, RParen);

        // Closing paren of pattern
        match_token!(self.lexer, RParen);

        // Validate constraints
        if min_hops > 1 {
            return Err(self.lexer.format_error(&format!(
                "{fn_name} does not support a minimal length different from 0 or 1"
            )));
        }
        if has_edge_filter {
            return Err(self.lexer.format_error(&format!(
                "filters on relationships in {fn_name} are not allowed"
            )));
        }

        // Determine actual source and dest based on direction
        let (actual_src, actual_dst) = if is_incoming && !is_outgoing {
            // <-[]-  means dst is the pattern's left node
            (dst_var, src_var)
        } else {
            (src_var, dst_var)
        };

        Ok(tree!(
            ExprIR::ShortestPath(Box::new(ShortestPathInfo {
                rel_types,
                min_hops,
                max_hops,
                directed,
            })),
            tree!(ExprIR::Variable(actual_src)),
            tree!(ExprIR::Variable(actual_dst))
        ))
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primary_expr(
        &mut self,
        allow_pattern_predicate: bool,
    ) -> Result<(DynTree<ExprIR<Arc<String>>>, bool), String> {
        match self.lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Case),
                ..
            } => Ok((self.parse_case_expression()?, false)),
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::All | Keyword::Any | Keyword::None | Keyword::Single),
                ..
            } => {
                // Quantifier only when followed by '(': all(x IN list WHERE ...).
                // Otherwise treat the keyword as a plain variable name.
                let state = self.save_state();
                self.lexer.next();
                let is_quantifier = matches!(self.lexer.current(), Ok(Token::LParen));
                self.restore_state(state);
                if is_quantifier {
                    Ok((self.parse_quantifier_expr(allow_pattern_predicate)?, false))
                } else {
                    let ident = self.parse_ident()?;
                    Ok((tree!(ExprIR::Variable(ident)), false))
                }
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::Null),
                ..
            } => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::Null)), false))
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::True),
                ..
            } => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::Bool(true))), false))
            }
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::False),
                ..
            } => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::Bool(false))), false))
            }
            Token::IdentifierOrKeyword { .. } => {
                let state = self.save_state();
                let ident = self.parse_dotted_ident()?;
                if optional_match_token!(self.lexer, LParen) {
                    // The dotted name is a call target: a `Lib.func` UDF name
                    // is capped as a whole, not per dot-separated component.
                    validate_identifier_len(ident.as_str(), "Function name")?;

                    // reduce(acc = init, var IN list | expr)
                    if ident.eq_ignore_ascii_case("reduce") {
                        return Ok((self.parse_reduce_expr(allow_pattern_predicate)?, false));
                    }

                    // shortestPath((a)-[*]->(b)) or allShortestPaths((a)-[*]->(b))
                    if ident.eq_ignore_ascii_case("shortestPath") {
                        return Ok((self.parse_shortest_path_expr()?, false));
                    }
                    if ident.eq_ignore_ascii_case("allShortestPaths") {
                        return Err(self.lexer.format_error(
                            "FalkorDB support allShortestPaths only in match clauses",
                        ));
                    }

                    let func = get_functions()
                        .get(&ident, &FnType::Function)
                        .or_else(|_| {
                            get_functions().get(
                                &ident,
                                &FnType::Aggregation {
                                    initial: Value::Null,
                                    finalizer: None,
                                    batch_agg: None,
                                },
                            )
                        })?;

                    let distinct = optional_match_token!(self.lexer => Distinct);

                    if func.is_aggregate() {
                        if optional_match_token!(self.lexer, Star) {
                            if func.name != "count" {
                                return Err(self.lexer.format_error(
                                    "COUNT is the only function which can accept * as an argument",
                                ));
                            }
                            if distinct {
                                return Err(self
                                    .lexer
                                    .format_error("COUNT(DISTINCT *) is not supported"));
                            }
                            // Create args array like count(x) does
                            let mut args = vec![tree!(ExprIR::Constant(Value::Int(1)))]; // Dummy value for count(*)

                            if distinct {
                                args = vec![tree!(ExprIR::Distinct; args)];
                            }

                            args.push(tree!(ExprIR::Variable(Arc::new(String::from(
                                "__agg_order_by_placeholder__"
                            )))));

                            match_token!(self.lexer, RParen);
                            return Ok((tree!(ExprIR::FuncInvocation(func); args), false));
                        }

                        let mut args = self.parse_expression_list(
                            ExpressionListType::ZeroOrMoreClosedBy(RParen),
                            allow_pattern_predicate,
                        )?;
                        func.validate(args.len())?;

                        // Check for nested aggregate functions
                        for arg in &args {
                            if Self::find_aggregate_name(arg).is_some() {
                                return Err(self.lexer.format_error(
                                    "Can't use aggregate functions inside of aggregate functions",
                                ));
                            }
                        }

                        if distinct {
                            args = vec![tree!(ExprIR::Distinct; args)];
                        }
                        args.push(tree!(ExprIR::Variable(Arc::new(String::from(
                            "__agg_order_by_placeholder__"
                        )))));
                        return Ok((tree!(ExprIR::FuncInvocation(func); args), false));
                    }

                    let args = self.parse_expression_list(
                        ExpressionListType::ZeroOrMoreClosedBy(RParen),
                        allow_pattern_predicate,
                    )?;
                    func.validate(args.len())?;
                    if distinct && args.is_empty() {
                        return Err(self.lexer.format_error(
                            "DISTINCT can only be used with function calls that have arguments",
                        ));
                    }
                    return Ok((tree!(ExprIR::FuncInvocation(func); args), false));
                }
                self.restore_state(state);
                let ident = self.parse_ident()?;
                Ok((tree!(ExprIR::Variable(ident)), false))
            }
            Token::Parameter(param) => {
                self.lexer.next();
                Ok((tree!(ExprIR::Parameter(param)), false))
            }
            Token::Integer(i) => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::Int(i))), false))
            }
            Token::Float(f) => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::Float(f))), false))
            }
            Token::String(s) => {
                self.lexer.next();
                Ok((tree!(ExprIR::Constant(Value::String(s))), false))
            }
            Token::LBrace => {
                self.lexer.next();
                self.parse_list_literal_or_comprehension(allow_pattern_predicate)
            }
            Token::LBracket => Ok((self.parse_map()?, false)),
            Token::LParen => {
                let checkpoint = self.save_state();
                // Try to detect pattern predicate: (ident)--(...) or (ident)<--(...)
                if allow_pattern_predicate
                    && let Ok(pattern) = self.parse_pattern(&Keyword::Match)
                    && !pattern.relationships().is_empty()
                {
                    return Ok((tree!(ExprIR::Pattern(Box::new(pattern))), false));
                }

                // Not a pattern predicate - restore and parse normally
                self.restore_state(checkpoint);
                self.lexer.next(); // re-consume LParen
                let expr = tree!(ExprIR::Paren);
                Ok((expr, true))
            }
            token => Err(self.lexer.format_error(&format!("Invalid input {token:?}"))),
        }
    }

    // match one of those kind [..4], [4..], [4..5], [6]
    fn parse_list_operator_expression(
        &mut self,
        mut lhs: DynTree<ExprIR<Arc<String>>>,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let from = self.parse_expr(false);
        if optional_match_token!(self.lexer, DotDot) {
            let to = self.parse_expr(false);
            match_token!(self.lexer, RBrace);
            lhs = tree!(
                ExprIR::GetElements,
                lhs,
                from.unwrap_or_else(|_| tree!(ExprIR::Constant(Value::Int(0)))),
                to.unwrap_or_else(|_| tree!(ExprIR::Constant(Value::Int(i64::MAX))))
            );
        } else {
            match_token!(self.lexer, RBrace);
            lhs = tree!(ExprIR::GetElement, lhs, from?);
        }

        Ok(lhs)
    }

    fn parse_property_lookup(
        &mut self,
        expr: DynTree<ExprIR<Arc<String>>>,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let ident = self.parse_ident()?;
        Ok(tree!(ExprIR::Property(ident), expr))
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cognitive_complexity)]
    fn parse_expr(
        &mut self,
        allow_pattern_predicate: bool,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let mut stack = vec![(0, None::<DynTree<ExprIR<Arc<String>>>>)];
        while let Some((current, res)) = stack.pop() {
            let Some(res) = res else {
                if current < 3 || (current > 3 && current < 9) || current == 10 {
                    stack.push((current, None));
                    stack.push((current + 1, None));
                } else if current == 3 {
                    // Not
                    let mut not_count = 0;
                    while let Token::IdentifierOrKeyword {
                        keyword: Some(Keyword::Not),
                        ..
                    } = self.lexer.current()?
                    {
                        self.lexer.next();
                        not_count += 1;
                    }
                    let res = if not_count % 2 == 1 {
                        Some(tree!(ExprIR::Not))
                    } else {
                        None
                    };
                    stack.push((current, res));
                    stack.push((current + 1, None));
                } else if current == 9 {
                    // unary add or subtract
                    //
                    // Signs chain, so `- -5` and `--5` are both `5`. Only the
                    // parity of the minus signs matters, the same way `NOT`
                    // folds above.
                    let mut is_negate = false;
                    while let Ok(token @ (Token::Plus | Token::Dash)) = self.lexer.current() {
                        is_negate ^= token == Token::Dash;
                        self.lexer.next();
                    }

                    // Handle integer overflow with negation
                    if is_negate && let Err(err) = self.lexer.current() {
                        return Err(err.replace("Integer overflow '", "Integer overflow '-"));
                    }

                    let res = if is_negate {
                        Some(tree!(ExprIR::Negate))
                    } else {
                        None
                    };
                    stack.push((current, res));
                    stack.push((current + 1, None));
                } else {
                    // primary expression
                    let (res, recurse) = self.parse_primary_expr(allow_pattern_predicate)?;
                    if recurse {
                        stack.push((current, Some(res)));
                        stack.push((0, None));
                        continue;
                    }
                    parse_expr_return!(stack, res);
                }
                continue;
            };
            match current {
                0 => {
                    // Or
                    parse_operators!(self, stack, res, current, Token::IdentifierOrKeyword { keyword: Some(Keyword::Or), .. } => Or);
                }
                1 => {
                    // Xor
                    parse_operators!(self, stack, res, current, Token::IdentifierOrKeyword { keyword: Some(Keyword::Xor), .. } => Xor);
                }
                2 => {
                    // And
                    parse_operators!(self, stack, res, current, Token::IdentifierOrKeyword { keyword: Some(Keyword::And), .. } => And);
                }
                3 => {
                    // Not
                    parse_expr_return!(stack, res);
                }
                4 => {
                    // Comparison with chained-range desugaring.
                    // Cypher `a < b <= c` means `a < b AND b <= c`.
                    let mut res = res;
                    let cmp_op = match self.lexer.current()? {
                        Token::Equal => Some(ExprIR::Eq),
                        Token::NotEqual => Some(ExprIR::Neq),
                        Token::LessThan => Some(ExprIR::Lt),
                        Token::LessThanOrEqual => Some(ExprIR::Le),
                        Token::GreaterThan => Some(ExprIR::Gt),
                        Token::GreaterThanOrEqual => Some(ExprIR::Ge),
                        _ => None,
                    };
                    if let Some(op) = cmp_op {
                        self.lexer.next();
                        // Detect chained comparison: res is a comparison node,
                        // or an And wrapping previous chain steps.
                        let last_cmp_is_ordering = match res.root().data() {
                            ExprIR::Lt
                            | ExprIR::Le
                            | ExprIR::Gt
                            | ExprIR::Ge
                            | ExprIR::Eq
                            | ExprIR::Neq => true,
                            ExprIR::And => res.root().children().last().is_some_and(|c| {
                                matches!(
                                    c.data(),
                                    ExprIR::Lt
                                        | ExprIR::Le
                                        | ExprIR::Gt
                                        | ExprIR::Ge
                                        | ExprIR::Eq
                                        | ExprIR::Neq
                                )
                            }),
                            _ => false,
                        };
                        if last_cmp_is_ordering {
                            // Clone the shared middle operand (last child of
                            // the latest comparison node in the chain).
                            let last_cmp_node = if matches!(res.root().data(), ExprIR::And) {
                                res.root().children().last().unwrap()
                            } else {
                                res.root()
                            };
                            let middle_clone: DynTree<ExprIR<Arc<String>>> =
                                last_cmp_node.children().last().unwrap().clone_as_tree();
                            // Wrap in And if not already from a prior step.
                            if !matches!(res.root().data(), ExprIR::And) {
                                res = tree!(ExprIR::And, res);
                            }
                            let new_cmp = tree!(op, middle_clone);
                            stack.push((current, Some(res)));
                            stack.push((current, Some(new_cmp)));
                        } else {
                            res = tree!(op, res);
                            stack.push((current, Some(res)));
                        }
                        stack.push((current + 1, None));
                        continue;
                    }

                    match &mut stack.last_mut() {
                        Some((_, Some(expr))) => {
                            expr.root_mut().push_child_tree(res);
                        }
                        Some((_, expr)) => {
                            *expr = Some(res);
                        }
                        _ => return Ok(res),
                    }
                }
                5 => {
                    // String, List, Null predicates
                    let mut res = res;
                    match self.lexer.current()? {
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::In),
                            ..
                        } => {
                            self.lexer.next();
                            res = tree!(ExprIR::In, res);
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Starts),
                            ..
                        } => {
                            self.lexer.next();
                            match_token!(self.lexer => With);
                            res = tree!(
                                ExprIR::FuncInvocation(
                                    get_functions().get("starts_with", &FnType::Internal)?,
                                ),
                                res
                            );
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Ends),
                            ..
                        } => {
                            self.lexer.next();
                            match_token!(self.lexer => With);
                            res = tree!(
                                ExprIR::FuncInvocation(
                                    get_functions().get("ends_with", &FnType::Internal)?,
                                ),
                                res
                            );
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Contains),
                            ..
                        } => {
                            self.lexer.next();
                            res = tree!(
                                ExprIR::FuncInvocation(
                                    get_functions().get("contains", &FnType::Internal)?,
                                ),
                                res
                            );
                        }
                        Token::RegexMatches => {
                            self.lexer.next();
                            res = tree!(
                                ExprIR::FuncInvocation(
                                    get_functions().get("regex_matches", &FnType::Internal)?,
                                ),
                                res
                            );
                        }
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Is),
                            ..
                        } => {
                            while optional_match_token!(self.lexer => Is) {
                                let is_not = tree!(ExprIR::Constant(Value::Bool(
                                    optional_match_token!(self.lexer => Not)
                                )));
                                match_token!(self.lexer => Null);
                                res = tree!(
                                    ExprIR::FuncInvocation(
                                        get_functions().get("is_null", &FnType::Internal)?
                                    ),
                                    is_not,
                                    res
                                );
                            }
                            parse_expr_return!(stack, res);
                            continue;
                        }
                        // Negated predicates: peek after NOT to decide
                        // whether it negates a predicate keyword.
                        //
                        // For recognized predicates we use a double-push:
                        //   1. Push Not() wrapper onto the stack.
                        //   2. Set `res` to the inner predicate with `lhs`
                        //      already attached.
                        // When the rhs returns from levels 6+, it becomes a
                        // child of the predicate, which in turn becomes a
                        // child of Not():
                        //   `x NOT IN [1,2]` → Not(In(x, [1,2]))
                        //
                        // Bare NOT (e.g. `u.v NOT NULL`) produces a binary
                        // Not(lhs, rhs) which the binder rejects.
                        Token::IdentifierOrKeyword {
                            keyword: Some(Keyword::Not),
                            ..
                        } => {
                            self.lexer.next();
                            match self.lexer.current()? {
                                // x NOT IN [1, 2, 3]
                                Token::IdentifierOrKeyword {
                                    keyword: Some(Keyword::In),
                                    ..
                                } => {
                                    self.lexer.next();
                                    stack.push((current, Some(tree!(ExprIR::Not))));
                                    res = tree!(ExprIR::In, res);
                                }
                                // name NOT STARTS WITH 'A'
                                Token::IdentifierOrKeyword {
                                    keyword: Some(Keyword::Starts),
                                    ..
                                } => {
                                    self.lexer.next();
                                    match_token!(self.lexer => With);
                                    stack.push((current, Some(tree!(ExprIR::Not))));
                                    res = tree!(
                                        ExprIR::FuncInvocation(
                                            get_functions()
                                                .get("starts_with", &FnType::Internal)?,
                                        ),
                                        res
                                    );
                                }
                                // name NOT ENDS WITH 'z'
                                Token::IdentifierOrKeyword {
                                    keyword: Some(Keyword::Ends),
                                    ..
                                } => {
                                    self.lexer.next();
                                    match_token!(self.lexer => With);
                                    stack.push((current, Some(tree!(ExprIR::Not))));
                                    res = tree!(
                                        ExprIR::FuncInvocation(
                                            get_functions().get("ends_with", &FnType::Internal)?,
                                        ),
                                        res
                                    );
                                }
                                // name NOT CONTAINS 'foo'
                                Token::IdentifierOrKeyword {
                                    keyword: Some(Keyword::Contains),
                                    ..
                                } => {
                                    self.lexer.next();
                                    stack.push((current, Some(tree!(ExprIR::Not))));
                                    res = tree!(
                                        ExprIR::FuncInvocation(
                                            get_functions().get("contains", &FnType::Internal)?,
                                        ),
                                        res
                                    );
                                }
                                // Bare NOT without a recognized predicate keyword
                                // (e.g. `u.v NOT NULL` instead of `u.v IS NOT NULL`).
                                _ => {
                                    return Err(self
                                        .lexer
                                        .format_error("Invalid usage of 'NOT' filter"));
                                }
                            }
                        }
                        _ => {
                            parse_expr_return!(stack, res);
                            continue;
                        }
                    }
                    stack.push((current, Some(res)));
                    stack.push((current + 1, None));
                }
                6 => {
                    // Add, Sub
                    parse_operators!(self, stack, res, current, Token::Plus => Add, Token::Dash => Sub);
                }
                7 => {
                    // Mul, Div, Modulo
                    parse_operators!(self, stack, res, current, Token::Star => Mul, Token::Slash => Div, Token::Modulo => Modulo);
                }
                8 => {
                    // Power
                    parse_operators!(self, stack, res, current, Token::Power => Pow);
                }
                9 => {
                    // unary add or subtract
                    if matches!(res.root().data(), ExprIR::Negate)
                        && matches!(
                            res.root().child(0).data(),
                            ExprIR::Constant(Value::Int(i64::MIN))
                        )
                    {
                        let res = tree!(ExprIR::Constant(Value::Int(i64::MIN)));
                        parse_expr_return!(stack, res);
                        continue;
                    } else if matches!(res.root().data(), ExprIR::Constant(Value::Int(i64::MIN))) {
                        // This case should not happen with proper error handling
                        // i64::MIN without negation means the literal was i64::MAX + 1
                        return Err(format!(
                            "Integer overflow '{}'",
                            9_223_372_036_854_775_808_u64
                        ));
                    }
                    parse_expr_return!(stack, res);
                }
                10 => {
                    // None arithmetic operators
                    let mut res = res;
                    loop {
                        match self.lexer.current()? {
                            Token::LBrace => {
                                self.lexer.next();
                                res = self.parse_list_operator_expression(res)?;
                            }
                            Token::Dot => {
                                self.lexer.next();
                                res = self.parse_property_lookup(res)?;
                            }
                            Token::LBracket => {
                                self.lexer.next();
                                res = self.parse_map_projection(res)?;
                            }
                            _ => break,
                        }
                    }
                    if self.lexer.current()? == Token::Colon {
                        let labels = tree!(ExprIR::List; self.parse_labels()?.into_iter().map(|l| tree!(ExprIR::Constant(Value::String(l)))));
                        res = tree!(
                            ExprIR::FuncInvocation(
                                get_functions().get("hasLabels", &FnType::Function)?
                            ),
                            res,
                            labels
                        );
                    }
                    parse_expr_return!(stack, res);
                }
                11 => {
                    // primary expression
                    let mut res = res;
                    if matches!(res.root().data(), ExprIR::Paren) {
                        match_token!(self.lexer, RParen);
                        if matches!(res.root().child(0).data(), ExprIR::Paren) {
                            res.root_mut().child_mut(0).take_out();
                        }
                    } else if matches!(res.root().data(), ExprIR::List) {
                        if optional_match_token!(self.lexer, Comma) {
                            stack.push((current, Some(res)));
                            stack.push((0, None));
                            continue;
                        }
                        match_token!(self.lexer, RBrace);
                    }
                    parse_expr_return!(stack, res);
                }
                _ => unreachable!(),
            }
        }
        unreachable!()
    }

    /// Match a dot separator in index property references (e.g. `n.prop`).
    /// Handles the edge case where `.1` is lexed as a float token instead of
    /// dot + integer, producing "expected a property name" in that case.
    fn match_dot_property_separator(&mut self) -> Result<(), String> {
        match self.lexer.current()? {
            Token::Dot => {
                self.lexer.next();
                Ok(())
            }
            Token::Float(_) if self.lexer.current_str().starts_with('.') => {
                Err(self.lexer.format_error(&format!(
                    "Invalid input '{}': expected a property name",
                    &self.lexer.current_str()[1..],
                )))
            }
            _ => Err(self.lexer.format_error(&format!(
                "Invalid input '{}': expected '.'",
                self.lexer.current_str(),
            ))),
        }
    }

    fn parse_ident(&mut self) -> Result<Arc<String>, String> {
        match self.lexer.current() {
            Ok(Token::IdentifierOrKeyword { ident: id, .. }) => {
                validate_identifier_len(id.as_str(), "Identifier")?;
                self.lexer.next();
                Ok(id)
            }
            _ => Err(self.lexer.format_error(&format!(
                "Invalid input '{}': expected an identifier",
                self.lexer.current_str(),
            ))),
        }
    }

    fn parse_property_name(&mut self) -> Result<Arc<String>, String> {
        match self.lexer.current() {
            Ok(Token::IdentifierOrKeyword { ident: id, .. }) => {
                validate_identifier_len(id.as_str(), "Property name")?;
                self.lexer.next();
                Ok(id)
            }
            _ => Err(self.lexer.format_error(&format!(
                "Invalid input '{}': expected a property name",
                self.lexer.current_str(),
            ))),
        }
    }

    fn parse_named_exprs(
        &mut self,
        must_alias: bool,
    ) -> Result<Vec<(Arc<String>, QueryExpr<Arc<String>>)>, String> {
        let mut named_exprs = Vec::new();
        loop {
            let pos = self.lexer.pos(false);
            let expr = Arc::new(self.parse_expr(true)?);
            if let Token::IdentifierOrKeyword {
                keyword: Some(Keyword::As),
                ..
            } = self.lexer.current()?
            {
                self.lexer.next();
                let ident = self.parse_ident()?;
                named_exprs.push((ident, expr));
            } else if let ExprIR::Variable(id) = expr.root().data() {
                named_exprs.push((id.clone(), expr));
            } else {
                if must_alias {
                    return Err(self
                        .lexer
                        .format_error("WITH clause projections must be aliased"));
                }
                named_exprs.push((
                    Arc::new(String::from(&self.lexer.str[pos..self.lexer.pos(true)])),
                    expr,
                ));
            }
            match self.lexer.current()? {
                Token::Comma => self.lexer.next(),
                _ => return Ok(named_exprs),
            }
        }
    }

    fn parse_expression_list(
        &mut self,
        expression_list_type: ExpressionListType,
        allow_pattern_predicate: bool,
    ) -> Result<Vec<DynTree<ExprIR<Arc<String>>>>, String> {
        let mut exprs = Vec::new();
        while !expression_list_type.is_end_token(&self.lexer.current()?) {
            exprs.push(self.parse_expr(allow_pattern_predicate)?);
            match self.lexer.current()? {
                Token::Comma => self.lexer.next(),
                _ => break,
            }
        }

        if let ExpressionListType::ZeroOrMoreClosedBy(token) = expression_list_type {
            if self.lexer.current()? == token {
                self.lexer.next();
            } else {
                return Err(self.lexer.format_error(&format!("Invalid input {token:?}")));
            }
        }
        Ok(exprs)
    }

    /// Parses the contents after an opening `[` bracket.
    ///
    /// Uses backtracking to distinguish between:
    /// 1. List comprehension: `[var IN list WHERE cond | expr]`
    /// 2. Named pattern comprehension: `[var = (pattern) WHERE cond | expr]`
    /// 3. Unnamed pattern comprehension: `[(pattern) WHERE cond | expr]`
    /// 4. List literal: `[1, 2, 3]` or `[]`
    ///
    /// Returns `(tree, recurse)` where `recurse` indicates the caller must
    /// continue parsing comma-separated elements for a list literal.
    fn parse_list_literal_or_comprehension(
        &mut self,
        allow_pattern_predicate: bool,
    ) -> Result<(DynTree<ExprIR<Arc<String>>>, bool), String> {
        let saved = self.save_state();

        // 1) Try list comprehension: [var IN ...]
        if let Ok(var) = self.parse_ident()
            && optional_match_token!(self.lexer => In)
        {
            return Ok((
                self.parse_list_comprehension(var, allow_pattern_predicate)?,
                false,
            ));
        }
        self.restore_state(saved);

        // 2) Try named pattern comprehension: [var = (pattern) ... | expr]
        if let Ok(var) = self.parse_ident()
            && optional_match_token!(self.lexer, Equal)
            && self.lexer.current()? == Token::LParen
            && let Ok(result) = self.parse_pattern_comprehension(Some(var), allow_pattern_predicate)
        {
            self.reject_aggregate(&result)?;
            return Ok((result, false));
        }
        self.restore_state(saved);

        // 3) Try unnamed pattern comprehension: [(pattern) ... | expr]
        if self.lexer.current()? == Token::LParen {
            if let Ok(result) = self.parse_pattern_comprehension(None, allow_pattern_predicate) {
                self.reject_aggregate(&result)?;
                return Ok((result, false));
            }
            self.restore_state(saved);
        }

        // 4) Default: list literal
        Ok((
            tree!(ExprIR::List),
            !optional_match_token!(self.lexer, RBrace),
        ))
    }

    fn parse_list_comprehension(
        &mut self,
        var: Arc<String>,
        allow_pattern_predicate: bool,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        // var and 'IN' already parsed
        let list_expr = self.parse_expr(allow_pattern_predicate)?;

        let condition = if optional_match_token!(self.lexer => Where) {
            Some(self.parse_expr(allow_pattern_predicate)?)
        } else {
            None
        };

        let expression = if optional_match_token!(self.lexer, Pipe) {
            let expr = self.parse_expr(allow_pattern_predicate)?;
            // Aggregation functions are not allowed inside list comprehensions.
            if let Some(func) = Self::find_aggregate_name(&expr) {
                return Err(self
                    .lexer
                    .format_error(&format!("Invalid use of aggregating function '{func}'")));
            }
            Some(expr)
        } else {
            None
        };

        match_token!(self.lexer, RBrace);

        Ok(tree!(
            ExprIR::ListComprehension(var.clone()),
            list_expr,
            condition.unwrap_or_else(|| tree!(ExprIR::Constant(Value::Bool(true)))),
            expression.map_or_else(|| Ok::<_, String>(tree!(ExprIR::Variable(var))), Ok)?
        ))
    }

    /// Parses a pattern comprehension after the opening `[`.
    ///
    /// Grammar: `[` (path_var `=`)? relationship_pattern (`WHERE` cond)? `|` expr `]`
    ///
    /// Collects the graph pattern into a `PatternComprehension` node
    /// for execution by the runtime.
    fn parse_pattern_comprehension(
        &mut self,
        path_var: Option<Arc<String>>,
        allow_pattern_predicate: bool,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let first_node = self.parse_node_pattern()?;

        // Must have at least one relationship
        if !matches!(self.lexer.current()?, Token::Dash | Token::LessThan) {
            return Err("Expected relationship pattern".into());
        }

        let mut graph = QueryGraph::default();
        graph.add_node(first_node.clone());

        let mut left = first_node;
        while matches!(self.lexer.current()?, Token::Dash | Token::LessThan) {
            let (rel, right) = self.parse_relationship_pattern(left, &Keyword::Match)?;
            graph.add_node(right.clone());
            graph.add_relationship(rel);
            left = right;
        }

        if let Some(pv) = path_var {
            graph.add_path(Arc::new(QueryPath::new(pv, vec![])));
        }

        // Optional WHERE
        let condition = if optional_match_token!(self.lexer => Where) {
            self.parse_expr(allow_pattern_predicate)?
        } else {
            tree!(ExprIR::Constant(Value::Bool(true)))
        };

        // Pipe + result expression
        match_token!(self.lexer, Pipe);
        let result_expr = self.parse_expr(allow_pattern_predicate)?;
        match_token!(self.lexer, RBrace);

        Ok(tree!(
            ExprIR::PatternComprehension(Box::new(graph)),
            condition,
            result_expr
        ))
    }

    /// Parses an inline property map, preserving "Unknown function" errors
    /// while replacing other parse errors with a generic inlined-properties message.
    fn parse_inline_properties(&mut self) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        self.parse_map().map_err(|e| {
            if e.starts_with("Unknown function") {
                e
            } else {
                String::from("Encountered unhandled type in inlined properties.")
            }
        })
    }

    fn parse_node_pattern(&mut self) -> Result<Arc<QueryNode<Arc<String>, Arc<String>>>, String> {
        match_token!(self.lexer, LParen);
        let alias = if let Ok(id) = self.parse_ident() {
            id
        } else {
            let name = Arc::new(format!("_anon_{}", self.anon_counter));
            self.anon_counter += 1;
            name
        };
        let labels = self.parse_labels()?;
        let attrs = if let Token::Parameter(param) = self.lexer.current()? {
            self.lexer.next();
            tree!(ExprIR::Parameter(param))
        } else if self.lexer.current()? == Token::LBracket {
            self.parse_inline_properties()?
        } else {
            tree!(ExprIR::Map)
        };
        match_token!(self.lexer, RParen);
        Ok(Arc::new(QueryNode::new(alias, labels, Arc::new(attrs))))
    }

    #[allow(clippy::too_many_lines)]
    fn parse_relationship_pattern(
        &mut self,
        src: Arc<QueryNode<Arc<String>, Arc<String>>>,
        clause: &Keyword,
    ) -> Result<
        (
            Arc<QueryRelationship<Arc<String>, Arc<String>, Arc<String>>>,
            Arc<QueryNode<Arc<String>, Arc<String>>>,
        ),
        String,
    > {
        let is_incoming = optional_match_token!(self.lexer, LessThan);
        match_token!(self.lexer, Dash);
        let has_details = optional_match_token!(self.lexer, LBrace);
        let (alias, types, attrs, var_len) = if has_details {
            let alias = if let Ok(id) = self.parse_ident() {
                id
            } else {
                let name = Arc::new(format!("_anon_{}", self.anon_counter));
                self.anon_counter += 1;
                name
            };
            let mut types = HashSet::new();
            if optional_match_token!(self.lexer, Colon) {
                loop {
                    types.insert(self.parse_ident()?);
                    let pipe = optional_match_token!(self.lexer, Pipe);
                    let colon = optional_match_token!(self.lexer, Colon);
                    if pipe || colon {
                        continue;
                    }
                    break;
                }
            }
            let var_len = if optional_match_token!(self.lexer, Star) {
                let start = if let Token::Integer(i) = self.lexer.current()? {
                    self.lexer.next();
                    Some(i)
                } else {
                    None
                };
                if optional_match_token!(self.lexer, DotDot) {
                    let end = if let Token::Integer(i) = self.lexer.current()? {
                        self.lexer.next();
                        Some(i)
                    } else {
                        None
                    };
                    // [*a..b] → min=a (default 1), max=b (default None=infinity)
                    Some((start.unwrap_or(1) as u32, end.map(|e| e as u32)))
                } else if let Some(exact) = start {
                    // [*N] → exact N hops
                    Some((exact as u32, Some(exact as u32)))
                } else {
                    // [*] → 1..infinity
                    Some((1, None))
                }
            } else {
                None
            };
            if let Some((min, Some(max))) = var_len
                && min > max
            {
                return Err(self.lexer.format_error(
                    "Variable length path, maximum number of hops must be greater or equal to minimum number of hops.",
                ));
            }
            // Collapse [*1..1] / [*1] to a fixed single-hop edge so the
            // binding is a single Relationship value, matching FalkorDB.
            let var_len = if var_len == Some((1, Some(1))) {
                None
            } else {
                var_len
            };
            if var_len.is_some() && matches!(clause, Keyword::Create | Keyword::Merge) {
                let clause_name = if *clause == Keyword::Create {
                    "CREATE"
                } else {
                    "MERGE"
                };
                return Err(format!(
                    "Variable length relationships cannot be used in {clause_name} patterns."
                ));
            }
            let attrs = if let Token::Parameter(param) = self.lexer.current()? {
                self.lexer.next();
                tree!(ExprIR::Parameter(param))
            } else if self.lexer.current()? == Token::LBracket {
                self.parse_inline_properties()?
            } else {
                tree!(ExprIR::Map)
            };
            match_token!(self.lexer, RBrace);
            (alias, types.into_iter().collect(), attrs, var_len)
        } else {
            let name = Arc::new(format!("_anon_{}", self.anon_counter));
            self.anon_counter += 1;
            (name, vec![], tree!(ExprIR::Map), None)
        };
        match_token!(self.lexer, Dash);
        let is_outgoing = optional_match_token!(self.lexer, GreaterThan);
        let dst = self.parse_node_pattern()?;
        let (min_hops, max_hops) = match var_len {
            Some((min, max)) => (Some(min), max),
            None => (None, None),
        };
        let relationship = match (is_incoming, is_outgoing) {
            (true, true) | (false, false) => {
                if *clause == Keyword::Create {
                    return Err(self
                        .lexer
                        .format_error("Only directed relationships are supported in CREATE"));
                }
                QueryRelationship::new(
                    alias,
                    types,
                    Arc::new(attrs),
                    src,
                    dst.clone(),
                    true,
                    min_hops,
                    max_hops,
                )
            }
            (true, false) => QueryRelationship::new(
                alias,
                types,
                Arc::new(attrs),
                dst.clone(),
                src,
                false,
                min_hops,
                max_hops,
            ),
            (false, true) => QueryRelationship::new(
                alias,
                types,
                Arc::new(attrs),
                src,
                dst.clone(),
                false,
                min_hops,
                max_hops,
            ),
        };
        Ok((Arc::new(relationship), dst))
    }

    fn parse_labels(&mut self) -> Result<OrderSet<Arc<String>>, String> {
        let mut labels = OrderSet::default();
        while self.lexer.current()? == Token::Colon {
            self.lexer.next();
            labels.insert(self.parse_ident()?);
        }
        Ok(labels)
    }

    fn parse_map(&mut self) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let mut attrs = Vec::new();
        if self.lexer.current()? == Token::LBracket {
            self.lexer.next();
        } else {
            return Ok(tree!(ExprIR::Map));
        }

        if self.lexer.current() == Ok(Token::RBracket) {
            self.lexer.next();
            return Ok(tree!(ExprIR::Map));
        }

        loop {
            let key = self.parse_ident()?;
            match_token!(self.lexer, Colon);
            let value = self.parse_expr(false)?;
            attrs.push(tree!(ExprIR::Constant(Value::String(key)), value));

            match self.lexer.current()? {
                Token::Comma => self.lexer.next(),
                Token::RBracket => {
                    self.lexer.next();
                    return Ok(tree!(ExprIR::Map ; attrs));
                }
                token => {
                    return Err(self.lexer.format_error(&format!("Invalid input {token:?}")));
                }
            }
        }
    }

    fn parse_map_projection(
        &mut self,
        base: DynTree<ExprIR<Arc<String>>>,
    ) -> Result<DynTree<ExprIR<Arc<String>>>, String> {
        let mut items = vec![base];

        // Empty projection: expr {}
        if self.lexer.current()? == Token::RBracket {
            self.lexer.next();
            return Ok(tree!(ExprIR::MapProjection ; items));
        }

        loop {
            if self.lexer.current()? == Token::Dot {
                self.lexer.next();
                if optional_match_token!(self.lexer, Star) {
                    // .* - all properties
                    items.push(tree!(ExprIR::MapProjection));
                } else {
                    // .property - property shorthand
                    let prop_name = self.parse_property_name()?;
                    items.push(tree!(ExprIR::Property(prop_name)));
                }
            } else {
                // key: expr  or  variable shorthand
                let ident = self.parse_ident()?;
                if optional_match_token!(self.lexer, Colon) {
                    let value = self.parse_expr(false)?;
                    items.push(tree!(ExprIR::Constant(Value::String(ident)), value));
                } else {
                    // variable shorthand: name -> name: name
                    items.push(tree!(
                        ExprIR::Constant(Value::String(ident.clone())),
                        tree!(ExprIR::Variable(ident))
                    ));
                }
            }

            match self.lexer.current()? {
                Token::Comma => {
                    self.lexer.next();
                }
                Token::RBracket => {
                    self.lexer.next();
                    break;
                }
                _ => {
                    return Err(self.lexer.format_error(&format!(
                        "Invalid input '{}': expected ':', ',' or '}}'",
                        self.lexer.current_str()
                    )));
                }
            }
        }

        Ok(tree!(ExprIR::MapProjection ; items))
    }

    fn parse_orderby(&mut self) -> Result<Vec<(QueryExpr<Arc<String>>, bool)>, String> {
        match_token!(self.lexer => By);
        let mut orderby = vec![];
        loop {
            let expr = Arc::new(self.parse_expr(false)?);
            let is_ascending = optional_match_token!(self.lexer => Asc)
                || optional_match_token!(self.lexer => Ascending);
            let is_descending = !is_ascending
                && (optional_match_token!(self.lexer => Desc)
                    || optional_match_token!(self.lexer => Descending));
            orderby.push((expr, is_descending));
            if !optional_match_token!(self.lexer, Comma) {
                break;
            }
        }
        Ok(orderby)
    }

    fn parse_set_clause(&mut self) -> Result<QueryIR<Arc<String>>, String> {
        let mut set_items = vec![];
        self.parse_set_items(&mut set_items)?;

        // Combine consecutive SET clauses into one
        while optional_match_token!(self.lexer => Set) {
            self.parse_set_items(&mut set_items)?;
        }

        Ok(QueryIR::Set(set_items))
    }

    fn parse_set_items(
        &mut self,
        set_items: &mut Vec<SetItem<Arc<String>, Arc<String>>>,
    ) -> Result<(), String> {
        loop {
            let (mut expr, recurse) = self.parse_primary_expr(false)?;
            if recurse {
                expr = self.parse_expr(false)?;
                match_token!(self.lexer, RParen);
            }
            if self.lexer.current()? == Token::Dot {
                while self.lexer.current()? == Token::Dot {
                    self.lexer.next();
                    expr = self.parse_property_lookup(expr)?;
                }
                match_token!(self.lexer, Equal);
                let value = Arc::new(self.parse_expr(false)?);
                set_items.push(SetItem::Attribute {
                    target: Arc::new(expr),
                    value,
                    replace: false,
                });
            } else if self.lexer.current()? == Token::Colon {
                let ExprIR::Variable(id) = expr.root().data() else {
                    return Err(self
                        .lexer
                        .format_error("Cannot set labels on non-node expressions"));
                };
                set_items.push(SetItem::Label {
                    var: id.clone(),
                    labels: self.parse_labels()?,
                });
            } else {
                let equals = optional_match_token!(self.lexer, Equal);
                let plus_equals = if equals {
                    false
                } else {
                    match_token!(self.lexer, PlusEqual);
                    true
                };
                let value = Arc::new(self.parse_expr(false)?);
                set_items.push(SetItem::Attribute {
                    target: Arc::new(expr),
                    value,
                    replace: !plus_equals,
                });
            }

            if !optional_match_token!(self.lexer, Comma) {
                return Ok(());
            }
        }
    }

    fn parse_remove_clause(&mut self) -> Result<QueryIR<Arc<String>>, String> {
        let mut remove_items = vec![];
        self.parse_remove_items(&mut remove_items)?;

        // Combine consecutive REMOVE clauses into one
        while optional_match_token!(self.lexer => Remove) {
            self.parse_remove_items(&mut remove_items)?;
        }

        Ok(QueryIR::Remove(remove_items))
    }

    fn parse_remove_items(
        &mut self,
        remove_items: &mut Vec<QueryExpr<Arc<String>>>,
    ) -> Result<(), String> {
        loop {
            let (mut expr, recurse) = self.parse_primary_expr(false)?;
            if recurse {
                expr = self.parse_expr(false)?;
                match_token!(self.lexer, RParen);
            }
            if self.lexer.current()? == Token::Dot {
                while self.lexer.current()? == Token::Dot {
                    self.lexer.next();
                    expr = self.parse_property_lookup(expr)?;
                }
                remove_items.push(Arc::new(expr));
            } else if self.lexer.current()? == Token::Colon {
                expr = tree!(
                    ExprIR::FuncInvocation(get_functions().get("hasLabels", &FnType::Function)?),
                    expr,
                    tree!(ExprIR::List; self.parse_labels()?.into_iter().map(|l| tree!(ExprIR::Constant(Value::String(l)))))
                );
                remove_items.push(Arc::new(expr));
            } else {
                return Err(self
                    .lexer
                    .format_error(&format!("Invalid input '{}'", self.lexer.current_str())));
            }

            if !optional_match_token!(self.lexer, Comma) {
                return Ok(());
            }
        }
    }

    fn parse_foreach_clause(&mut self) -> Result<RawQueryIR, String> {
        // FOREACH ( var IN list_expr | body_clauses )
        match_token!(self.lexer, LParen);
        let var = self.parse_ident()?;
        match_token!(self.lexer => In);
        let list_expr = self.parse_expr(false)?;
        // Check for aggregate functions in the list expression
        if let Some(func) = Self::find_aggregate_name(&list_expr) {
            return Err(self
                .lexer
                .format_error(&format!("Invalid use of aggregating function '{func}'")));
        }
        match_token!(self.lexer, Pipe);
        // Parse body clauses: CREATE, MERGE, SET, DELETE, DETACH DELETE, REMOVE, FOREACH
        let mut body = Vec::new();
        loop {
            match self.lexer.current()? {
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Create),
                    ..
                } => {
                    self.lexer.next();
                    body.push(self.parse_create_clause()?);
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Merge),
                    ..
                } => {
                    self.lexer.next();
                    body.push(self.parse_merge_clause()?);
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Detach | Keyword::Delete),
                    ..
                } => {
                    let is_detach = optional_match_token!(self.lexer => Detach);
                    match_token!(self.lexer => Delete);
                    body.push(self.parse_delete_clause(is_detach)?);
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Set),
                    ..
                } => {
                    self.lexer.next();
                    body.push(self.parse_set_clause()?);
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Remove),
                    ..
                } => {
                    self.lexer.next();
                    body.push(self.parse_remove_clause()?);
                }
                Token::IdentifierOrKeyword {
                    keyword: Some(Keyword::Foreach),
                    ..
                } => {
                    self.lexer.next();
                    body.push(self.parse_foreach_clause()?);
                }
                _ => break,
            }
        }
        match_token!(self.lexer, RParen);
        if body.is_empty() {
            return Err(self
                .lexer
                .format_error("FOREACH body must contain at least one clause"));
        }
        Ok(QueryIR::ForEach {
            list: Arc::new(list_expr),
            var,
            body,
        })
    }

    /// Reject aggregation functions anywhere inside `tree`.
    ///
    /// Called on a fully-parsed pattern comprehension, after the backtracking
    /// caller has committed to that interpretation — validating inside
    /// `parse_pattern_comprehension` would let the caller swallow the error and
    /// retry the input as a list literal, masking it with a syntax error.
    fn reject_aggregate(
        &self,
        tree: &DynTree<ExprIR<Arc<String>>>,
    ) -> Result<(), String> {
        if let Some(func) = Self::find_aggregate_name(tree) {
            return Err(self
                .lexer
                .format_error(&format!("Invalid use of aggregating function '{func}'")));
        }
        Ok(())
    }

    fn find_aggregate_name(tree: &DynTree<ExprIR<Arc<String>>>) -> Option<&str> {
        use orx_tree::Dfs;
        for idx in tree.root().indices::<Dfs>() {
            if let ExprIR::FuncInvocation(func) = tree.node(idx).data()
                && func.is_aggregate()
            {
                return Some(&func.name);
            }
        }
        None
    }
}
