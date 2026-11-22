import FalkorTemporal.PadDefs

/-! Slice 4 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice4 : FourDigitsSlice 4 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
