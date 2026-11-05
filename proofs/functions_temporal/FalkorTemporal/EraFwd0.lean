import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_0 : allRange fwdOk 0 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_40000 : allRange fwdOk 40000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_80000 : allRange fwdOk 80000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_120000 : allRange fwdOk 120000 10000 = true := by decide +kernel

end FalkorTemporal
