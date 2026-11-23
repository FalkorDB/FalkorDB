/-
# planner_build

The planner (`graph/src/planner/mod.rs`, `ast.rs`): clause plans, stitching into whole queries,
MATCH lowering, filter and expression planning, hop ordering and renaming.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import PlannerBuild.Core
import PlannerBuild.Stitch
import PlannerBuild.Clauses
import PlannerBuild.Match
import PlannerBuild.Expr
import PlannerBuild.Anchor
import PlannerBuild.Decomp
import PlannerBuild.AndOf
import PlannerBuild.ToPlan
import PlannerBuild.Filter
import PlannerBuild.FilterPlan
import PlannerBuild.Nested
import PlannerBuild.Nested2
import PlannerBuild.Misc
import PlannerBuild.PlanMatch
import PlannerBuild.Lift
import PlannerBuild.PlanMatch2
import PlannerBuild.RemoveVerified
import PlannerBuild.HopOrder
import PlannerBuild.Fmt
import PlannerBuild.Rename
import PlannerBuild.Ast
import PlannerBuild.Ast2
import PlannerBuild.PathBind
import PlannerBuild.PlanStack
