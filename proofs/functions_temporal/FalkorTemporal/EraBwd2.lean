import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_20000 : allRange bwdOk 20000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_60000 : allRange bwdOk 60000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_100000 : allRange bwdOk 100000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_140000 : allRange bwdOk 140000 8800 = true := by decide +kernel

end FalkorTemporal
