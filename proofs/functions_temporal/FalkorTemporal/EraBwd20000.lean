import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_20000_0 : allRange bwdOk 20000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_20000_1 : allRange bwdOk 22500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_20000_2 : allRange bwdOk 25000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_20000_3 : allRange bwdOk 27500 2500 = true := by decide +kernel

theorem bwd_20000 : allRange bwdOk 20000 10000 = true := allRange_append bwdOk 20000 7500 2500 (allRange_append bwdOk 20000 5000 2500 (allRange_append bwdOk 20000 2500 2500 (bwd_20000_0) bwd_20000_1) bwd_20000_2) bwd_20000_3

end FalkorTemporal
