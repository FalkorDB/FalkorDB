import RedisLayer.BufferedIO
/-!
# `src/serializers/mod.rs` — v19 graph header and schema blocks

Encoders emit the logical `Writer` calls (`BufferedIO.W`); decoders consume them. By
`BufferedIO.roundtrip` the reader hands back exactly those calls, so a token-level round
trip is a byte-level one. Strings are byte lists; `lossy` is `String::from_utf8_lossy`,
of which we use only that it is the identity on valid UTF-8 (every Rust `String`).

| here | there (`origin/main`) |
| --- | --- |
| `nullTerm`, `strip`       | `null_terminated` `:139,:244`, `strip_null_terminator` `:252` (and the inline copy in `Header::decode` `:166-170`) |
| `encH`, `decH`            | `Header::encode` `:135-161`, `Header::decode` `:164-199` |
| `fromGraph`               | `Header::from_graph` `:212-232` |
| `encField`, `decField`    | field loop of `encode_schema_index_block` `:382-421`, `decode_index_field` `:621-699` |
| `encIdx`, `encCons`       | `encode_schema_index_block` `:337-422`, `encode_constraint_block` `:439-464` |
| `decEntry`                | `decode_schema_entry` `:529-619` |
| `encSchema`, `decSchema`  | `Schema::encode` `:260-331`, `Schema::decode` `:466-527` |
| `VKS`, `DS`               | `VirtualKeyState::new/clear` `:38-48`, `DecodeState::new/clear` `:100-115` |
| `schemaFromGraph`         | `Schema::from_graph` `:702-718` |
-/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

/-- What we use of `String::from_utf8_lossy`. -/
structure Utf8 where
  valid : Bytes → Prop
  lossy : Bytes → Bytes
  lossy_valid : ∀ s, valid s → lossy s = s

variable (U : Utf8)

def nullTerm (s : Bytes) : Bytes := s ++ [0]

def strip (b : Bytes) : Bytes :=
  if b.getLast? = some 0 then U.lossy b.dropLast else U.lossy b

/-- A name survives `null_terminated` → `strip_null_terminator`, including one that
itself ends in NUL (only the added terminator is removed). -/
theorem strip_nullTerm (s : Bytes) (h : U.valid s) : strip U (nullTerm s) = s := by
  simp [strip, nullTerm, U.lossy_valid s h]

/-- A C-written name without terminator is taken whole. -/
theorem strip_noterm (b : Bytes) (h : b.getLast? ≠ some 0) : strip U b = U.lossy b := by
  simp [strip, h]

/-! ## Token decoders -/

abbrev Dec (α : Type) := List W → Option (α × List W)

def Dec.bind {α β} (p : Dec α) (f : α → Dec β) : Dec β := fun ts =>
  match p ts with
  | some (a, ts) => f a ts
  | none => none

def pure' {α} (a : α) : Dec α := fun ts => some (a, ts)

def rdU : Dec Nat
  | .u n :: ts => some (n, ts)
  | _ => none
def rdD : Dec Nat
  | .d n :: ts => some (n, ts)
  | _ => none
def rdB : Dec Bytes
  | .buf b :: ts => some (b, ts)
  | _ => none

/-- `for _ in 0..n { … }` collecting into a `Vec`. -/
theorem Dec.bind_apply {α β} (p : Dec α) (f : α → Dec β) (ts : List W) :
    Dec.bind p f ts = match p ts with
      | some (a, ts) => f a ts
      | none => none := rfl
theorem Dec.bind_some {α β} (p : Dec α) (f : α → Dec β) (ts ts' : List W) (a : α)
    (h : p ts = some (a, ts')) : Dec.bind p f ts = f a ts' := by
  simp [Dec.bind, h]
@[simp] theorem rdU_u (n : Nat) (ts : List W) : rdU (.u n :: ts) = some (n, ts) := rfl
@[simp] theorem rdD_d (n : Nat) (ts : List W) : rdD (.d n :: ts) = some (n, ts) := rfl
@[simp] theorem rdB_buf (b : Bytes) (ts : List W) : rdB (.buf b :: ts) = some (b, ts) := rfl
@[simp] theorem pure'_apply {α} (a : α) (ts : List W) : pure' a ts = some (a, ts) := rfl

def rdMany {α} (p : Dec α) : Nat → Dec (List α)
  | 0 => pure' []
  | n+1 => Dec.bind p fun a => Dec.bind (rdMany p n) fun as => pure' (a :: as)

theorem rdMany_enc {α} (p : Dec α) (e : α → List W) (xs : List α)
    (hp : ∀ x ∈ xs, ∀ ts, p (e x ++ ts) = some (x, ts)) (ts : List W) :
    rdMany p xs.length ((xs.map e).flatten ++ ts) = some (xs, ts) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [rdMany, Dec.bind, List.length_cons, List.map_cons, List.flatten_cons,
      List.append_assoc]
    rw [hp x (by simp)]
    simp only
    rw [ih (fun y hy => hp y (by simp [hy]))]
    rfl

/-! ## Header -/

structure Header where
  name : Bytes
  nc : Nat
  ec : Nat
  dnc : Nat
  dec : Nat
  lc : Nat
  rc : Nat
  multiEdge : List Bool
  kc : Nat
  deriving DecidableEq, Repr

def b2u (b : Bool) : Nat := if b then 1 else 0

def encH (h : Header) : List W :=
  [.buf (nullTerm h.name), .u h.nc, .u h.ec, .u h.dnc, .u h.dec, .u h.lc, .u h.rc]
    ++ (h.multiEdge.map fun b => [W.u (b2u b)]).flatten ++ [.u h.kc]

def decH : Dec Header :=
  Dec.bind rdB fun nb =>
  Dec.bind rdU fun nc => Dec.bind rdU fun ec => Dec.bind rdU fun dnc =>
  Dec.bind rdU fun dec => Dec.bind rdU fun lc => Dec.bind rdU fun rc =>
  Dec.bind (rdMany (Dec.bind rdU fun f => pure' (decide (f ≠ 0))) rc) fun me =>
  Dec.bind rdU fun kc =>
  pure' ⟨strip U nb, nc, ec, dnc, dec, lc, rc, me, kc⟩

/-- **Header round trip**, given the invariant that there is one multi-edge flag per
relationship type (which `fromGraph` establishes). -/
theorem decH_encH (h : Header) (hv : U.valid h.name) (hme : h.multiEdge.length = h.rc)
    (ts : List W) : decH U (encH h ++ ts) = some (h, ts) := by
  obtain ⟨name, nc, ec, dnc, dec, lc, rc, me, kc⟩ := h
  simp only at hv hme
  subst hme
  have hm := rdMany_enc (Dec.bind rdU fun f => pure' (decide (f ≠ 0))) (fun b => [W.u (b2u b)]) me
    (by intro b _ ts; cases b <;> simp [Dec.bind, rdU, pure', b2u]) (W.u kc :: ts)
  simp only [decH, encH, Dec.bind, rdB, rdU, List.cons_append, List.append_assoc, List.nil_append]
  rw [hm]
  simp [pure', strip_nullTerm U name hv]

/-- `Header::from_graph`: the flags come from the same tensor list as the count. -/
def fromGraph (name : Bytes) (nc ec dnc dec lc : Nat) (multi : List Bool) (kc : Nat) : Header :=
  ⟨name, nc, ec, dnc, dec, lc, multi.length, multi, kc⟩

theorem fromGraph_inv (name : Bytes) (nc ec dnc dec lc : Nat) (multi : List Bool) (kc : Nat) :
    (fromGraph name nc ec dnc dec lc multi kc).multiEdge.length
      = (fromGraph name nc ec dnc dec lc multi kc).rc := rfl

/-- A flag word other than 0/1 (a C writer is free to) decodes as `true`. -/
theorem decH_flag_nonzero (n : Nat) (hn : n ≠ 0) (ts : List W) :
    (Dec.bind rdU fun f => pure' (decide (f ≠ 0))) (.u n :: ts) = some (true, ts) := by
  simp [Dec.bind, rdU, pure', hn]

/-! ## Global decode/encode state (`:26-116`) -/

structure VKS (V G : Type) where
  vkeyMap : List (Bytes × V)
  graphVkeys : List (Bytes × G)

def VKS.new {V G} : VKS V G := ⟨[], []⟩
def VKS.clear {V G} (_ : VKS V G) : VKS V G := ⟨[], []⟩
theorem VKS.clear_eq_new {V G} (s : VKS V G) : s.clear = VKS.new := rfl

structure DS (P H F : Type) where
  pending : List (Bytes × P)
  placeholders : List (Bytes × H)
  finalized : List (Bytes × F)
  metaKeys : List Bytes

def DS.new {P H F} : DS P H F := ⟨[], [], [], []⟩
def DS.clear {P H F} (_ : DS P H F) : DS P H F := ⟨[], [], [], []⟩
theorem DS.clear_eq_new {P H F} (s : DS P H F) : s.clear = DS.new := rfl
theorem DS.new_empty {P H F} : (DS.new : DS P H F).metaKeys = [] ∧ (DS.new : DS P H F).pending = [] :=
  ⟨rfl, rfl⟩

end RedisLayer.Serial
