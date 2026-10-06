import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_30000 : allRange fwdOk 30000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_70000 : allRange fwdOk 70000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_110000 : allRange fwdOk 110000 10000 = true := by decide +kernel

end FalkorTemporal
