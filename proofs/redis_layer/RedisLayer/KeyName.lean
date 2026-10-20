import RedisLayer.GraphCore
/-!
# The key a write is replicated under, as bytes (#2653) — gap closed 2026-10-20

`GraphCore.cGraphKey` models the key a new graph is *stored* under as bytes, and
`GraphCore.Lossy` keeps UTF-8 decoding abstract. What reaches replication and
`WATCH` signalling is `key_name: Arc<str>`, a `String`, so a non-UTF-8 key goes through
`from_utf8_lossy`. This file gives a concrete lossy decoder and states the property.

| here | there (fb3d516d0) |
| --- | --- |
| `decF`, `lossy` | `String::from_utf8_lossy` (Rust `core`: each maximal invalid subpart becomes one U+FFFD) |
| `encCp`, `utf8Bytes` | `str::as_bytes` of the resulting `String` |
| `storedKey`  | existing graph: the addressed key (`query.rs:72`); created graph: `c_graph_key` bytes (`query.rs:144`, `graph_core.rs:119-130`) |
| `rustReplicatedKey` | `key_name` = `key_str.to_string()` (`query.rs:73`; `RedisString: Display` is `to_string_lossy`, redismodule-rs `redismodule.rs:323-327`) or `c_graph_name` (`query.rs:113-114`), then `signal_modified_key(ctx, key_name.as_bytes())` (`graph_core.rs:1591`) and `buf.replicate(.., key_name.as_bytes())` (`graph_core.rs:1612`) |
| `cReplicatedKey` | C replicates with the command's own `RedisModuleString` (bytes unchanged) |
-/
namespace RedisLayer.KeyName
open RedisLayer.GraphCore

def isCont (b : Nat) : Bool := 0x80 ≤ b && b ≤ 0xBF

/-- `from_utf8_lossy` to code points, with fuel (one byte consumed per step at least). -/
def decF : Nat → Bytes → List Nat
  | 0, _ => []
  | _, [] => []
  | f + 1, b :: rest =>
    if b < 0x80 then b :: decF f rest
    else if 0xC2 ≤ b ∧ b ≤ 0xDF then
      match rest with
      | c :: r => if isCont c then ((b % 32) * 64 + c % 64) :: decF f r else 0xFFFD :: decF f rest
      | [] => [0xFFFD]
    else if 0xE0 ≤ b ∧ b ≤ 0xEF then
      let lo := if b = 0xE0 then 0xA0 else 0x80
      let hi := if b = 0xED then 0x9F else 0xBF
      match rest with
      | c1 :: r1 =>
        if lo ≤ c1 ∧ c1 ≤ hi then
          match r1 with
          | c2 :: r2 =>
            if isCont c2 then ((b % 16) * 4096 + (c1 % 64) * 64 + c2 % 64) :: decF f r2
            else 0xFFFD :: decF f r1
          | [] => [0xFFFD]
        else 0xFFFD :: decF f rest
      | [] => [0xFFFD]
    else if 0xF0 ≤ b ∧ b ≤ 0xF4 then
      let lo := if b = 0xF0 then 0x90 else 0x80
      let hi := if b = 0xF4 then 0x8F else 0xBF
      match rest with
      | c1 :: r1 =>
        if lo ≤ c1 ∧ c1 ≤ hi then
          match r1 with
          | c2 :: r2 =>
            if isCont c2 then
              match r2 with
              | c3 :: r3 =>
                if isCont c3 then
                  ((b % 8) * 262144 + (c1 % 64) * 4096 + (c2 % 64) * 64 + c3 % 64) :: decF f r3
                else 0xFFFD :: decF f r2
              | [] => [0xFFFD]
            else 0xFFFD :: decF f r1
          | [] => [0xFFFD]
        else 0xFFFD :: decF f rest
      | [] => [0xFFFD]
    else 0xFFFD :: decF f rest

def lossy (k : Bytes) : List Nat := decF k.length k

/-- UTF-8 encoding of one scalar value. -/
def encCp (c : Nat) : Bytes :=
  if c < 0x80 then [c]
  else if c < 0x800 then [0xC0 + c / 64, 0x80 + c % 64]
  else if c < 0x10000 then [0xE0 + c / 4096, 0x80 + (c / 64) % 64, 0x80 + c % 64]
  else [0xF0 + c / 262144, 0x80 + (c / 4096) % 64, 0x80 + (c / 64) % 64, 0x80 + c % 64]

def utf8Bytes (cs : List Nat) : Bytes := cs.flatMap encCp

/-- The key the graph lives at (NUL-free keys: `c_graph_key` is the identity on them,
`GraphCore.upToNulB_split`). -/
def storedKey (k : Bytes) : Bytes := cGraphKey k
def rustReplicatedKey (k : Bytes) : Bytes := utf8Bytes (lossy k)
def cReplicatedKey (k : Bytes) : Bytes := k

/-- **Replication names the key the graph lives at, byte for byte**: for every NUL-free
key, the key written to the replication stream / signalled to `WATCH` equals the key the
graph is stored under on the primary. -/
def ReplicatedKeyIsStored (repl : Bytes → Bytes) : Prop :=
  ∀ k : Bytes, 0 ∉ k → repl k = storedKey k

theorem c_ok : ReplicatedKeyIsStored cReplicatedKey := by
  intro k h; simp [cReplicatedKey, storedKey, cGraphKey, (upToNulB_split k).2 h]

theorem decF_ascii (k : Bytes) (h : ∀ b ∈ k, b < 0x80) : decF k.length k = k := by
  induction k with
  | nil => rfl
  | cons b rest ih =>
    have hb : b < 0x80 := h b (by simp)
    simp [decF, hb, ih (fun x hx => h x (by simp [hx]))]

/-- ASCII keys (the common case) are kept. -/
theorem ascii_kept (k : Bytes) (h : ∀ b ∈ k, b < 0x80) : rustReplicatedKey k = k := by
  unfold rustReplicatedKey lossy; rw [decF_ascii k h]
  induction k with
  | nil => rfl
  | cons b rest ih =>
    have hb : b < 0x80 := h b (by simp)
    simp [utf8Bytes, encCp, hb] at ih ⊢
    exact ih (fun x hx => h x (by simp [hx]))

/-- The live repro keys: `\xc3(bin` and `\xff\xfe`. -/
def kBin : Bytes := [0xc3, 0x28, 0x62, 0x69, 0x6e]
def kFF : Bytes := [0xff, 0xfe]

/-- **BUG #2653**: an invalid-UTF-8 key is replicated with U+FFFD (`EF BF BD`) in place
of each invalid subpart, so the replica writes a different key. Live (primary 6650 →
replica 6652, Rust main): keys `\xc3(bin2`, `\xff\xfd` arrive as `\xef\xbf\xbd(bin2`,
`\xef\xbf\xbd\xef\xbf\xbd`; C's replica has the primary's keys. -/
theorem bug2653_nonutf8_key_mangled :
    rustReplicatedKey kBin = [0xef, 0xbf, 0xbd, 0x28, 0x62, 0x69, 0x6e] ∧
    rustReplicatedKey kFF = [0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd] ∧
    storedKey kBin = kBin ∧ storedKey kFF = kFF ∧
    ¬ ReplicatedKeyIsStored rustReplicatedKey := by
  refine ⟨by decide, by decide, by decide, by decide, fun h => ?_⟩
  have := h kFF (by decide); revert this; decide

/-- A valid multi-byte key survives (sanity check of the decoder): `é` = `C3 A9`. -/
theorem valid_two_byte_kept : rustReplicatedKey [0xc3, 0xa9] = [0xc3, 0xa9] := by decide

end RedisLayer.KeyName
