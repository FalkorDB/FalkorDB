import FalkorParserGrammar.BoundedDefs

/-! Chunk 9 of the bounded checks over `lists 3` (see `BoundedDefs.lean`). -/
namespace FalkorParserGrammar

set_option maxRecDepth 100000 in
theorem okAt_chunk9 : (chunk (lists 3) 9).all okAt = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem balOK_chunk9 : (chunk (lists 3) 9).all balOK = true := by decide +kernel

end FalkorParserGrammar
