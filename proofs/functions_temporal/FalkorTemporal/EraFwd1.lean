import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days). Evaluated by the
kernel (`decide +kernel`: no `native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_10000 : allRange fwdOk 10000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_50000 : allRange fwdOk 50000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_90000 : allRange fwdOk 90000 10000 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_130000 : allRange fwdOk 130000 10000 = true := by decide +kernel

end FalkorTemporal
