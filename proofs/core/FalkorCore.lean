import FalkorCore.Order
import FalkorCore.Laws
import FalkorCore.Int64
import FalkorCore.Float
import FalkorCore.Value
import FalkorCore.Range
/-!
# FalkorCore — the shared foundations of the Lean models (PILOT)

One model of what several projects used to copy:

| module | contents | was in |
| --- | --- | --- |
| `Order` | `cmpI`/`cmpN`/`cmpB`, `cmp3` | value_order `Model.lean` |
| `Laws`  | comparator laws `Lw`, `thn`, `lexA`, `cmp3_lw` | value_order `Laws.lean` |
| `Int64` | `I64MIN`/`I64MAX`, `InI64`, `wrap`, `wrapping_*`, `satI64`, `w` | value_order, expr_semantics, value_math, functions_str_list |
| `Range` | `rangeLen`, `rangeList`, `rangeIter`/`rangeIterRel` (`RangeIter::next` after #2899, debug/release), `rangeIter_of_inRange` | functions_str_list, expr_semantics |
| `Float` | `F` (NaN sign/payload, ±0, integral, fractional, ±inf), `fpcmp`, `cmpF`, `Flag`, `roundF` (`i as f64`), `bits64`/`bits32`, `asI64` | value_order `Model.lean` |
| `Value` | the engine `Value` as `V F` (all 16 variants), `order`, `name` | value_math `Value.lean` |

Projects use it through a Lake path dependency (`[[require]] name = "falkor-core"
path = "../core"`) and `open FalkorCore`.
-/
