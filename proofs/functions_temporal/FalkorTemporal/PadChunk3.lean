import FalkorTemporal.PadDefs

/-! Slice 3 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice3 : FourDigitsSlice 3 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
