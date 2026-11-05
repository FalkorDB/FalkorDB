import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_0 : allRange bwdOk 0 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_40000 : allRange bwdOk 40000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_80000 : allRange bwdOk 80000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_120000 : allRange bwdOk 120000 10000 = true := by decide +kernel

end FalkorTemporal
