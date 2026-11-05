import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_30000 : allRange bwdOk 30000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_70000 : allRange bwdOk 70000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_110000 : allRange bwdOk 110000 10000 = true := by decide +kernel

end FalkorTemporal
