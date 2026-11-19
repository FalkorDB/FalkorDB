import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem bwd_130000_0 : allRange bwdOk 130000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_130000_1 : allRange bwdOk 132500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_130000_2 : allRange bwdOk 135000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem bwd_130000_3 : allRange bwdOk 137500 2500 = true := by decide +kernel

theorem bwd_130000 : allRange bwdOk 130000 10000 = true := allRange_append bwdOk 130000 7500 2500 (allRange_append bwdOk 130000 5000 2500 (allRange_append bwdOk 130000 2500 2500 (bwd_130000_0) bwd_130000_1) bwd_130000_2) bwd_130000_3

end FalkorTemporal
