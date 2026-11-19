import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_70000_0 : allRange bwdOk 70000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_70000_1 : allRange bwdOk 72500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_70000_2 : allRange bwdOk 75000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_70000_3 : allRange bwdOk 77500 2500 = true := by decide +kernel

theorem bwd_70000 : allRange bwdOk 70000 10000 = true := allRange_append bwdOk 70000 7500 2500 (allRange_append bwdOk 70000 5000 2500 (allRange_append bwdOk 70000 2500 2500 (bwd_70000_0) bwd_70000_1) bwd_70000_2) bwd_70000_3

end FalkorTemporal
