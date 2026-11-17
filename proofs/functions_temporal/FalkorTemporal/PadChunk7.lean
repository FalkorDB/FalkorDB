import FalkorTemporal.PadDefs

/-! Slice 7 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice7 : FourDigitsSlice 7 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
