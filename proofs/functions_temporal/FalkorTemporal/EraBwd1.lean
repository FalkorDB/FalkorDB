import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_10000 : allRange bwdOk 10000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_50000 : allRange bwdOk 50000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_90000 : allRange bwdOk 90000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_130000 : allRange bwdOk 130000 10000 = true := by decide +kernel

end FalkorTemporal
