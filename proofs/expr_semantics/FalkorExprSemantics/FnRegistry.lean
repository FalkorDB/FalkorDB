/-!
# Function registry completeness against C (#2724) — gap closed 2026-10-20

`Functions::get` (`graph/src/runtime/functions/mod.rs:1048`) is proven elsewhere
(`value_math` `regGet_lower`: lookup by lowercase name). What it can find is whatever
`init_functions` (`mod.rs:1103-1120`) registered, and no property related that set to
the functions C offers. This file states that property.

| here | there |
| --- | --- |
| `rustNames` | every name `init_functions` registers, lowercased: the `cypher_fn!(funcs, "…"` and `funcs.add[_var_len\|_procedure]("…"` sites under `graph/src/runtime/functions/` at fb3d516d0, unchanged at 289c201fe (134 names, extracted mechanically) |
| `cPublic`   | every `AR_FuncDesc` in C `src/arithmetic/builtin_funcs.c` (origin/master d7cf86675, `git show`) *without* `.internal=true`, lowercased (104 names; the 40 internal ones — operators, `topath`, `list_comprehension`, … — are not callable by name, checked live) |
| `resolves`  | `Functions::get`'s built-in arm: `self.functions.get(name.to_lowercase())` |

MODELLED: the two name sets are data extracted from the sources, not derived from the
Rust code by proof; re-extract them when a function is added.
-/
namespace FalkorExprSemantics.FnRegistry

/-- Names registered by `init_functions` (lowercase). -/
def rustNames : List String := [
  "abs", "acos", "add", "algo.betweenness", "algo.bfs", "algo.harmoniccentrality",
  "algo.labelpropagation", "algo.maxflow", "algo.msf", "algo.pagerank", "algo.sppaths",
  "algo.sspaths", "algo.wcc", "asin", "atan", "atan2", "avg", "case", "ceil", "coalesce",
  "collect", "contains", "cos", "cot", "count", "date", "date.transaction",
  "db.constraints", "db.idx.fulltext.createnodeindex", "db.idx.fulltext.drop",
  "db.idx.fulltext.querynodes", "db.idx.fulltext.queryrelationships",
  "db.idx.vector.querynodes", "db.idx.vector.queryrelationships", "db.indexes",
  "db.labels", "db.meta.stats", "db.propertykeys", "db.relationshiptypes",
  "dbms.functions", "dbms.procedures", "degrees", "distance", "duration", "e", "endnode",
  "ends_with", "exists", "exp", "floor", "haslabels", "haversin", "head", "id", "indegree",
  "intern", "is_null", "isempty", "keys", "labels", "last", "left", "length", "list.dedup",
  "list.insert", "list.insertlistelements", "list.remove", "list.sort", "localdatetime",
  "localdatetime.transaction", "localtime", "localtime.transaction", "log", "log10",
  "ltrim", "max", "min", "nodes", "outdegree", "percentilecont", "percentiledisc", "pi",
  "point", "pow", "properties", "radians", "rand", "randomuuid", "range", "regex_matches",
  "relationships", "replace", "reverse", "right", "round", "rtrim", "sign", "sin", "size",
  "split", "sqrt", "startnode", "starts_with", "stdev", "stdevp", "string.join",
  "string.matchregex", "string.replaceregex", "substring", "sum", "tail", "tan",
  "timestamp", "toboolean", "tobooleanlist", "tobooleanornull", "tofloat", "tofloatlist",
  "tofloatornull", "tointeger", "tointegerlist", "tointegerornull", "tojson", "tolower",
  "tostring", "tostringlist", "tostringornull", "toupper", "trim", "type", "typeof",
  "vec.cosinedistance", "vec.euclideandistance", "vecf32"]

/-- C's user-callable built-in functions (lowercase). -/
def cPublic : List String := [
  "abs", "acos", "asin", "atan", "atan2", "avg", "ceil", "coalesce", "collect", "cos",
  "cot", "count", "date", "date.transaction", "degrees", "distance", "duration", "e",
  "endnode", "exists", "exp", "floor", "haslabels", "haversin", "head", "id", "indegree",
  "intern", "isempty", "keys", "labels", "last", "left", "length", "list.dedup",
  "list.insert", "list.insertlistelements", "list.remove", "list.sort", "localdatetime",
  "localdatetime.transaction", "localtime", "localtime.transaction", "log", "log10",
  "ltrim", "max", "min", "nodes", "outdegree", "percentilecont", "percentiledisc", "pi",
  "point", "pow", "prev", "properties", "radians", "rand", "randomuuid", "range",
  "relationships", "replace", "reverse", "right", "round", "rtrim", "sign", "sin", "size",
  "split", "sqrt", "startnode", "stdev", "stdevp", "string.join", "string.matchregex",
  "string.replaceregex", "substring", "sum", "tail", "tan", "timestamp", "toboolean",
  "tobooleanlist", "tobooleanornull", "tofloat", "tofloatlist", "tofloatornull",
  "tointeger", "tointegerlist", "tointegerornull", "tojson", "tolower", "tostring",
  "tostringlist", "tostringornull", "toupper", "trim", "type", "typeof",
  "vec.cosinedistance", "vec.euclideandistance", "vecf32"]

/-- `Functions::get`'s built-in lookup (names are stored and looked up lowercase). -/
def resolves (reg : List String) (name : String) : Bool := reg.contains name

/-- **The Rust function set covers C's**: every user-callable C function name resolves
in the Rust registry. -/
def CoversC (reg : List String) : Prop := ∀ n ∈ cPublic, resolves reg n = true

/-- Boolean form of "every C name except `prev` resolves", evaluated by the kernel. -/
def allButPrevResolve : Bool := cPublic.all (fun n => n == "prev" || resolves rustNames n)

theorem allButPrevResolve_true : allButPrevResolve = true := by decide

/-- `prev` is the only C function missing: every other one resolves. -/
theorem covers_all_but_prev : ∀ n ∈ cPublic, n ≠ "prev" → resolves rustNames n = true := by
  intro n hn hne
  have := List.all_eq_true.mp allButPrevResolve_true n hn
  simpa [hne] using this

/-- **BUG #2724**: `prev` is documented and registered in C (`_desc_prev`,
`builtin_funcs.c:1057`) but not in Rust. Live: Rust `RETURN prev(1)` → "Unknown function
'prev'"; C returns null. -/
theorem bug2724_prev_missing :
    "prev" ∈ cPublic ∧ resolves rustNames "prev" = false ∧ ¬ CoversC rustNames := by
  have h1 : "prev" ∈ cPublic := by decide
  have h2 : resolves rustNames "prev" = false := by decide
  exact ⟨h1, h2, fun h => by rw [h "prev" h1] at h2; cases h2⟩

end FalkorExprSemantics.FnRegistry
