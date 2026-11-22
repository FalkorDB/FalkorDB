import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_10000_0 : allRange bwdOk 10000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_10000_1 : allRange bwdOk 12500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_10000_2 : allRange bwdOk 15000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_10000_3 : allRange bwdOk 17500 2500 = true := by decide +kernel

theorem bwd_10000 : allRange bwdOk 10000 10000 = true := allRange_append bwdOk 10000 7500 2500 (allRange_append bwdOk 10000 5000 2500 (allRange_append bwdOk 10000 2500 2500 (bwd_10000_0) bwd_10000_1) bwd_10000_2) bwd_10000_3

end FalkorTemporal
