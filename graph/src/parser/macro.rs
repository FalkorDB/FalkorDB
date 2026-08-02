//! Cypher parser helper macros.
//!
//! This module defines internal macros used by [`crate::parser::cypher::Parser`]
//! to reduce repetitive token-matching and expression-tree construction logic.
//!
//! ## Macros
//!
//! ### `match_token!`
//! Mandatory token or keyword match. Consumes the token and advances the
//! lexer on success; returns a formatted parse error on mismatch.
//!
//! ```text
//! match_token!(lexer, LParen)       // expect Token::LParen
//! match_token!(lexer => Match)      // expect Keyword::Match
//! ```
//!
//! ### `optional_match_token!`
//! Optional token or keyword match. Returns `true` and advances if the
//! current token matches; returns `false` without advancing otherwise.
//!
//! ```text
//! if optional_match_token!(lexer, Comma) { ... }
//! if optional_match_token!(lexer => Where) { ... }
//! ```
//!
//! ### `parse_expr_return!`
//! Expression-stack return helper. After a sub-expression has been fully
//! parsed into `$res`, this macro either pushes it as a child of the
//! parent expression on the stack, stores it as the pending result, or
//! returns it if the stack is empty (i.e., we are at the top level).
//!
//! ### `parse_operators!`
//! Precedence-aware binary operator folding. Checks whether the current
//! token is one of the given operator tokens; if so, wraps the
//! left-hand operand in the corresponding `ExprIR` node and pushes a
//! new stack frame for the right-hand operand at the next higher
//! precedence level. If no operator matches, the result is folded back
//! into the parent stack frame.
//!
//! This stack-based approach avoids deep call-stack recursion for
//! heavily nested or chained binary expressions (e.g., `a+b+c+...`).

macro_rules! match_token {
    ($lexer:expr, $token:ident) => {
        match $lexer.current()? {
            Token::$token => {
                $lexer.next();
            }
            _ => {
                return Err($lexer.format_error(&format!(
                    // Return the display in error
                    "Invalid input '{}': expected {}",
                    $lexer.current_str(),
                    Token::$token
                )));
            }
        }
    };
    ($lexer:expr => $token:ident) => {
        match $lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::$token),
                ..
            } => {
                $lexer.next();
            }
            _ => {
                return Err($lexer.format_error(&format!(
                    "Invalid input '{}': expected {:?}",
                    $lexer.current_str(),
                    Keyword::$token
                )));
            }
        }
    };
    () => {};
}

macro_rules! optional_match_token {
    ($lexer:expr, $token:ident) => {
        match $lexer.current()? {
            Token::$token => {
                $lexer.next();
                true
            }
            _ => false,
        }
    };
    ($lexer:expr => $token:ident) => {
        match $lexer.current()? {
            Token::IdentifierOrKeyword {
                keyword: Some(Keyword::$token),
                ..
            } => {
                $lexer.next();
                true
            }
            _ => false,
        }
    };
    () => {};
}

#[macro_export]
macro_rules! parse_expr_return {
    ($stack:ident, $res:ident) => {
        match &mut $stack.last_mut() {
            Some((_, Some(expr))) => {
                expr.root_mut().push_child_tree($res);
            }
            Some((_, expr)) => {
                *expr = Some($res);
            }
            _ => return Ok($res),
        }
    };
}

#[macro_export]
macro_rules! parse_operators {
    ($self:ident, $stack:ident, $res:ident, $current:ident, $token:pat => $expr:ident) => {
        if let $token = $self.lexer.current()? {
            $self.lexer.next();
            let res = if matches!($res.root().data(), ExprIR::$expr) {
                $res
            } else {
                tree!(ExprIR::$expr, $res)
            };
            $stack.push(($current, Some(res)));
            $stack.push(($current + 1, None));
        } else {
            match &mut $stack.last_mut() {
                Some((_, Some(expr))) => {
                    expr.root_mut().push_child_tree($res);
                }
                Some((_, expr)) => {
                    *expr = Some($res);
                }
                _ => return Ok($res),
            }
        }
    };
    ($self:ident, $stack:ident, $res:ident, $current:ident, $($token:pat => $expr:ident),*) => {
        let mut res = $res;
        $(if let $token = $self.lexer.current()? {
            $self.lexer.next();
            if matches!(res.root().data(), ExprIR::$expr) {
            } else {
                res = tree!(ExprIR::$expr, res);
            };
            $stack.push(($current, Some(res)));
            $stack.push(($current + 1, None));
            continue;
        })*

        match &mut $stack.last_mut() {
            Some((_, Some(expr))) => {
                expr.root_mut().push_child_tree(res);
            }
            Some((_, expr)) => {
                *expr = Some(res);
            }
            _ => return Ok(res),
        }
    };
}
