import FalkorParserGrammar.BoundedDefs

/-! Chunk 4 of the bounded checks over `lists 3` (see `BoundedDefs.lean`). -/
namespace FalkorParserGrammar

set_option maxRecDepth 100000 in
theorem okAt_chunk4 : (chunk (lists 3) 4).all okAt = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem balOK_chunk4 : (chunk (lists 3) 4).all balOK = true := by decide +kernel

end FalkorParserGrammar
