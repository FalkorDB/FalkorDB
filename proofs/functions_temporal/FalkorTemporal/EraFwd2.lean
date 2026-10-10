import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_20000 : allRange fwdOk 20000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_60000 : allRange fwdOk 60000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_100000 : allRange fwdOk 100000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_140000 : allRange fwdOk 140000 6097 = true := by decide +kernel

end FalkorTemporal
