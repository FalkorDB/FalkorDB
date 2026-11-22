import FalkorTemporal.EraDefs
/-! Machine-checked enumeration over one 400-year era (146097 days), one 10000-point chunk per file, each in four 2500-point theorems (kernel memory grows with the range), so
Lake checks the chunks in parallel. Evaluated by the kernel (`decide +kernel`: no
`native_decide`, no `ofReduceBool` axiom). -/
namespace FalkorTemporal

set_option maxRecDepth 100000 in
theorem fwd_140000_0 : allRange fwdOk 140000 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_140000_1 : allRange fwdOk 142500 2500 = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem fwd_140000_2 : allRange fwdOk 145000 1097 = true := by decide +kernel

theorem fwd_140000 : allRange fwdOk 140000 6097 = true := allRange_append fwdOk 140000 5000 1097 (allRange_append fwdOk 140000 2500 2500 (fwd_140000_0) fwd_140000_1) fwd_140000_2

end FalkorTemporal
