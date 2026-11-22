import FalkorTemporal.PadDefs

/-! Slice 1 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice1 : FourDigitsSlice 1 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
