import FalkorTemporal.PadDefs

/-! Slice 2 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice2 : FourDigitsSlice 2 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
