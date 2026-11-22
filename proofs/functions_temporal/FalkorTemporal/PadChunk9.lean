import FalkorTemporal.PadDefs

/-! Slice 9 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice9 : FourDigitsSlice 9 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
