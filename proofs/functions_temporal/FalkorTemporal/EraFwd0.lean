import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days),
one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so Lake checks the chunks in parallel.
Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_0_0 : allRange fwdOk 0 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_0_1 : allRange fwdOk 2500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_0_2 : allRange fwdOk 5000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_0_3 : allRange fwdOk 7500 2500 = true := by decide +kernel

theorem fwd_0 : allRange fwdOk 0 10000 = true := allRange_append fwdOk 0 7500 2500 (allRange_append fwdOk 0 5000 2500 (allRange_append fwdOk 0 2500 2500 (fwd_0_0) fwd_0_1) fwd_0_2) fwd_0_3

end FalkorTemporal
