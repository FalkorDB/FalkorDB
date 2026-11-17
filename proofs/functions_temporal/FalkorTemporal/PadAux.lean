import FalkorTemporal.PadDefs
import FalkorTemporal.PadChunk0
import FalkorTemporal.PadChunk1
import FalkorTemporal.PadChunk2
import FalkorTemporal.PadChunk3
import FalkorTemporal.PadChunk4
import FalkorTemporal.PadChunk5
import FalkorTemporal.PadChunk6
import FalkorTemporal.PadChunk7
import FalkorTemporal.PadChunk8
import FalkorTemporal.PadChunk9

namespace FalkorTemporal.Fns

theorem fourDigits_aux :
    ∀ a, a < 100 → ∀ b, b < 100 → twoDigits a ++ twoDigits b = pad 4 (a * 100 + b) := by
  intro a ha b hb
  have hk : a / 10 < 10 := by omega
  have hs : FourDigitsSlice (a / 10) := by
    generalize hq : a / 10 = k at hk
    match k, hk with
    | 0, _ => exact fourDigits_slice0
    | 1, _ => exact fourDigits_slice1
    | 2, _ => exact fourDigits_slice2
    | 3, _ => exact fourDigits_slice3
    | 4, _ => exact fourDigits_slice4
    | 5, _ => exact fourDigits_slice5
    | 6, _ => exact fourDigits_slice6
    | 7, _ => exact fourDigits_slice7
    | 8, _ => exact fourDigits_slice8
    | 9, _ => exact fourDigits_slice9
  have := hs (a % 10) (by omega) b hb
  have e : 10 * (a / 10) + a % 10 = a := by omega
  rwa [e] at this

end FalkorTemporal.Fns
