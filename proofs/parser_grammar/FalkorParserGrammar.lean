/-
# parser_grammar

The Cypher parser (`graph/src/parser/cypher.rs`, `ast.rs`): the expression layer as a hand-written
precedence parser with a no-panic proof, and the clause layer proven sound against a transcription
of Cypher.g4 (`C/*`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorParserGrammar.Model
import FalkorParserGrammar.NoPanic
import FalkorParserGrammar.Grammar
import FalkorParserGrammar.Theorems
import FalkorParserGrammar.Bounded
import FalkorParserGrammar.Clauses
import FalkorParserGrammar.C.Core
import FalkorParserGrammar.C.Graph
import FalkorParserGrammar.C.Components
import FalkorParserGrammar.C.IR
import FalkorParserGrammar.C.Pattern
import FalkorParserGrammar.C.PatternG
import FalkorParserGrammar.C.PatternG2
import FalkorParserGrammar.C.Oracle0
import FalkorParserGrammar.C.PatternBugs
import FalkorParserGrammar.C.Clauses
import FalkorParserGrammar.C.ClausesNP
import FalkorParserGrammar.C.ClausesG
import FalkorParserGrammar.C.ClausesG2
import FalkorParserGrammar.C.ClausesG3
import FalkorParserGrammar.C.Index
import FalkorParserGrammar.C.ClauseBugs
import FalkorParserGrammar.C.ExprC
import FalkorParserGrammar.C.ExprC2
import FalkorParserGrammar.C.ExprC3
import FalkorParserGrammar.C.ExprC4
import FalkorParserGrammar.C.ExprC5
import FalkorParserGrammar.C.Primary
import FalkorParserGrammar.C.PrimaryG
import FalkorParserGrammar.C.Params
import FalkorParserGrammar.C.Fmt
import FalkorParserGrammar.C.ListGate
import FalkorParserGrammar.C.PredicateBugs
