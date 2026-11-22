import FalkorTemporal.PadDefs

/-! Slice 5 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice5 : FourDigitsSlice 5 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
