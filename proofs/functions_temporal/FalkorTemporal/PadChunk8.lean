import FalkorTemporal.PadDefs

/-! Slice 8 of `fourDigits_aux` (see `PadDefs.lean`). -/
namespace FalkorTemporal.Fns

theorem fourDigits_slice8 : FourDigitsSlice 8 := by
  unfold FourDigitsSlice; decide +kernel

end FalkorTemporal.Fns
