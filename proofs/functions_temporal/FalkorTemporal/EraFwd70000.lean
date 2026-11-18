import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_70000_0 : allRange fwdOk 70000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_70000_1 : allRange fwdOk 72500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_70000_2 : allRange fwdOk 75000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_70000_3 : allRange fwdOk 77500 2500 = true := by decide +kernel

theorem fwd_70000 : allRange fwdOk 70000 10000 = true := allRange_append fwdOk 70000 7500 2500 (allRange_append fwdOk 70000 5000 2500 (allRange_append fwdOk 70000 2500 2500 (fwd_70000_0) fwd_70000_1) fwd_70000_2) fwd_70000_3

end FalkorTemporal
