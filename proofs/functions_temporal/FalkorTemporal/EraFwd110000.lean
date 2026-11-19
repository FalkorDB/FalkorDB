import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_110000_0 : allRange fwdOk 110000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_110000_1 : allRange fwdOk 112500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_110000_2 : allRange fwdOk 115000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_110000_3 : allRange fwdOk 117500 2500 = true := by decide +kernel

theorem fwd_110000 : allRange fwdOk 110000 10000 = true := allRange_append fwdOk 110000 7500 2500 (allRange_append fwdOk 110000 5000 2500 (allRange_append fwdOk 110000 2500 2500 (fwd_110000_0) fwd_110000_1) fwd_110000_2) fwd_110000_3

end FalkorTemporal
