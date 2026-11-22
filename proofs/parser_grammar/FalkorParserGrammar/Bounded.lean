/-
# Bounded exhaustive agreement with the grammar
-/
import FalkorParserGrammar.Grammar
import FalkorParserGrammar.BoundedChunk0
import FalkorParserGrammar.BoundedChunk1
import FalkorParserGrammar.BoundedChunk2
import FalkorParserGrammar.BoundedChunk3
import FalkorParserGrammar.BoundedChunk4
import FalkorParserGrammar.BoundedChunk5
import FalkorParserGrammar.BoundedChunk6
import FalkorParserGrammar.BoundedChunk7
import FalkorParserGrammar.BoundedChunk8
import FalkorParserGrammar.BoundedChunk9

namespace FalkorParserGrammar

/-- Each of the ten chunks of `lists 3` passes `okAt`. -/
theorem okAt_chunks : ∀ i, i < 10 → (chunk (lists 3) i).all okAt = true := by
  intro i hi
  match i, hi with
  | 0, _ => exact okAt_chunk0
  | 1, _ => exact okAt_chunk1
  | 2, _ => exact okAt_chunk2
  | 3, _ => exact okAt_chunk3
  | 4, _ => exact okAt_chunk4
  | 5, _ => exact okAt_chunk5
  | 6, _ => exact okAt_chunk6
  | 7, _ => exact okAt_chunk7
  | 8, _ => exact okAt_chunk8
  | 9, _ => exact okAt_chunk9

/-- Each of the ten chunks of `lists 3` passes `balOK`. -/
theorem balOK_chunks : ∀ i, i < 10 → (chunk (lists 3) i).all balOK = true := by
  intro i hi
  match i, hi with
  | 0, _ => exact balOK_chunk0
  | 1, _ => exact balOK_chunk1
  | 2, _ => exact balOK_chunk2
  | 3, _ => exact balOK_chunk3
  | 4, _ => exact balOK_chunk4
  | 5, _ => exact balOK_chunk5
  | 6, _ => exact balOK_chunk6
  | 7, _ => exact balOK_chunk7
  | 8, _ => exact balOK_chunk8
  | 9, _ => exact balOK_chunk9

/-- **Bounded soundness and completeness**: on every one of the 9,724 token
lists of length ≤ 3 over `alpha`, the Rust model and the grammar accept the
same inputs with the same trees, outside the documented classes (`okAt`, `BoundedDefs.lean`). -/
theorem bounded_check : (lists 3).all okAt = true := all_lists3 okAt okAt_chunks

theorem bounded_agree :
    ∀ l ∈ lists 3, agreeB (parseAll 400 l) (G.all 200 l) = true ∨ known l = true := by
  intro l hl
  have := List.all_eq_true.mp bounded_check l hl
  simpa [okAt] using this

/-- **The fragment's `parse_expr` consumes balanced runs** on every token list of
length ≤ 3 over `alpha`. (The same check was run with `#eval` over all lists of
length ≤ 5 over `alpha` and of length ≤ 6 over the slice alphabet
`x 1 + ( ) [ ] ..`, with no failure.) -/
theorem bounded_bal : (lists 3).all balOK = true := all_lists3 balOK balOK_chunks

/-- The input `[1][(1..2]` (the pre-#3041 gate counterexample) no longer parses. -/
theorem slice_unclosed_paren_refused :
    ∃ e, parseExpr 400 [.lbrack, .int 1, .rbrack, .lbrack, .lparen, .int 1, .dotdot, .int 2, .rbrack] = .err e :=
  ⟨_, rfl⟩

end FalkorParserGrammar
