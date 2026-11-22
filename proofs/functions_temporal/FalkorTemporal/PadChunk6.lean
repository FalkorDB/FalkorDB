import FalkorTemporal.PadDefs

/-! Slice 6 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice6 : FourDigitsSlice 6 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
