import FalkorTemporal.PadDefs

/-! Slice 0 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice0 : FourDigitsSlice 0 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
