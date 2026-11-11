import RedisLayer.BufferedReader
/-!
# Write → read round trip of the v19 byte layer

`roundtrip`: whatever chunk source carries the writer's buffers (the RDB string buffers,
or the length-framed pipe), reading back the same sequence of `Writer` calls returns exactly
the values written, and leaves the reader at the end of the stream. The reader only ever
calls `ensure_available` before a tag, so this needs every value to sit wholly inside one
chunk and every blob sentinel to end its chunk — `writer_layout` below is that fact, and
it holds for every buffer size `B ≥ 9`.
-/
namespace RedisLayer.BufferedIO

section
variable (B : Nat)

theorem write_out (os : List Bytes) (b : Bytes) (x : W) :
    write B ⟨os, b⟩ x = ⟨os ++ (write B ⟨[], b⟩ x).out, (write B ⟨[], b⟩ x).buf⟩ := by
  by_cases h : inl B x <;> by_cases hb : b = [] <;>
    simp [write, h, accommodate, flush, push, hb] <;> split <;> simp_all

theorem fold_out (os : List Bytes) (b : Bytes) (xs : List W) :
    xs.foldl (write B) ⟨os, b⟩
      = ⟨os ++ (xs.foldl (write B) ⟨[], b⟩).out, (xs.foldl (write B) ⟨[], b⟩).buf⟩ := by
  induction xs generalizing os b with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl]
    rw [write_out B os b x, ih, ih (write B ⟨[], b⟩ x).out]
    simp

theorem finish_out (os : List Bytes) (b : Bytes) :
    finish ⟨os, b⟩ = os ++ finish ⟨[], b⟩ := by
  unfold finish flush; split <;> simp_all

theorem G_nil (b : Bytes) : G B b [] = if b = [] then [] else [b] := by
  simp [G, finish, flush]; split <;> simp_all

theorem G_cons (b : Bytes) (x : W) (xs : List W) :
    G B b (x :: xs) = (write B ⟨[], b⟩ x).out ++ G B (write B ⟨[], b⟩ x).buf xs := by
  unfold G
  simp only [List.foldl]
  generalize write B ⟨[], b⟩ x = w
  obtain ⟨o, bb⟩ := w
  rw [fold_out B o bb xs]
  generalize List.foldl (write B) ⟨[], bb⟩ xs = F
  obtain ⟨fo, fb⟩ := F
  simp only
  rw [finish_out (o ++ fo) fb, finish_out fo fb]
  simp

theorem record_len (x : W) (h9 : 9 ≤ B) (hi : inl B x = true) : (record x).length ≤ B := by
  cases x <;> simp_all [record, inl] <;> omega

theorem write_fit (b : Bytes) (x : W) (hi : inl B x = true) (h : b.length + (record x).length ≤ B) :
    write B ⟨[], b⟩ x = ⟨[], b ++ record x⟩ := by
  have : ¬ (b.length + (record x).length > B) := by omega
  simp only [write, hi, accommodate, push, ite_true, this, ite_false]

theorem write_spill (b : Bytes) (x : W) (hi : inl B x = true)
    (h : b.length + (record x).length > B) (hb : b ≠ []) :
    write B ⟨[], b⟩ x = ⟨[b], record x⟩ := by
  simp [write, hi, accommodate, push, flush, h, hb]

theorem write_blob_fit (b : Bytes) (x : W) (hi : inl B x = false) (h : b.length + 1 ≤ B) :
    write B ⟨[], b⟩ x = ⟨[b ++ [TYPE_BLOB], payload x], []⟩ := by
  have : ¬ (b.length + 1 > B) := by omega
  simp [write, hi, accommodate, push, flush, this]

theorem write_blob_spill (b : Bytes) (x : W) (hi : inl B x = false)
    (h : b.length + 1 > B) (hb : b ≠ []) :
    write B ⟨[], b⟩ x = ⟨[b, [TYPE_BLOB], payload x], []⟩ := by
  simp [write, hi, accommodate, push, flush, h, hb, TYPE_BLOB]

theorem blob_payload_ne (x : W) (h9 : 9 ≤ B) (hi : inl B x = false) : payload x ≠ [] := by
  cases x <;> simp_all [inl, payload]; intro h; subst h; simp at hi; omega

end

/-! ## Reader steps -/
section
variable {σ : Type} (S : Src σ)

theorem ensure_cons (c : Nat) (cs : Bytes) (s : σ) : ensure S ⟨c :: cs, s⟩ = .ok ⟨c :: cs, s⟩ := by
  simp [ensure]

theorem readW_inline (x : W) (r : Rd σ) (T : Bytes) (s : σ) (hx : x.ok)
    (he : ensure S r = .ok ⟨record x ++ T, s⟩) : readW S x r = .ok (x, ⟨T, s⟩) := by
  cases x with
  | u n =>
    simp [readW, readTag, he, Res.bind, record, TYPE_UNSIGNED, readBytes]
    simp [W.ok] at hx; simp [fromLE_le 8 n hx]
  | s i =>
    simp [readW, readTag, he, Res.bind, record, TYPE_SIGNED, readBytes]
    simp [W.ok] at hx; simp [fromLE_le 8 _ (toU_lt i hx), ofU_toU i hx]
  | d n =>
    simp [readW, readTag, he, Res.bind, record, TYPE_DOUBLE, readBytes]
    simp [W.ok] at hx; simp [fromLE_le 8 n hx]
  | f n =>
    simp [readW, readTag, he, Res.bind, record, TYPE_FLOAT, readBytes]
    simp [W.ok] at hx; simp [fromLE_le 4 n hx]
  | buf ds =>
    simp [W.ok] at hx
    simp [readW, he, Res.bind, record, TYPE_BYTES, readBytes, fromLE_le 8 _ hx]

theorem readW_blob (x : W) (r : Rd σ) (s : σ) (ds : Bytes) (s' : σ)
    (hx : ∃ d0, x = .buf d0)
    (he : ensure S r = .ok ⟨[TYPE_BLOB], s⟩) (hb : S.blob s = .ok (ds, s')) :
    readW S x r = .ok (.buf ds, ⟨[], s'⟩) := by
  obtain ⟨d0, rfl⟩ := hx
  simp [readW, he, Res.bind, hb, TYPE_BLOB, TYPE_BYTES]

theorem readSeq_cons_ok (x : W) (xs : List W) (r r' rf : Rd σ) (y : W) (ys : List W)
    (h1 : readW S x r = .ok (y, r')) (h2 : readSeq S xs r' = .ok (ys, rf)) :
    readSeq S (x :: xs) r = .ok (y :: ys, rf) := by
  simp [readSeq, h1, Res.bind, h2]

end

/-! ## The round trip -/
section
variable (B : Nat) {σ : Type} (S : Src σ) (cat : List Bytes → σ → σ)
  (hload : ∀ c cs t, c ≠ [] → c.length < 256^8 → S.load (cat (c :: cs) t) = .ok (c, cat cs t))
  (hblob : ∀ c cs t, c ≠ [] → c.length < 256^8 → S.blob (cat (c :: cs) t) = .ok (c, cat cs t))
  (h9 : 9 ≤ B) (hB : B < 256^8)
include hload hblob h9 hB

theorem load_head (c : Bytes) (cs : List Bytes) (t : σ) (hc : c ≠ []) (hl : c.length < 256^8) :
    ensure S ⟨[], cat (c :: cs) t⟩ = .ok ⟨c, cat cs t⟩ := by
  simp [ensure, hload c cs t hc hl, Res.bind]

omit hload hblob in
theorem G_bound (xs : List W) : ∀ (b : Bytes), b.length ≤ B → (∀ x ∈ xs, x.ok) →
    ∀ c ∈ G B b xs, c ≠ [] ∧ c.length < 256^8 := by
  induction xs with
  | nil =>
    intro b hb _ c hc
    rw [G_nil] at hc; split at hc
    · simp at hc
    · simp at hc; subst hc; exact ⟨by assumption, by omega⟩
  | cons x xs ih =>
    intro b hb hok c hc
    have hx : x.ok := hok x (by simp)
    have hxs : ∀ y ∈ xs, y.ok := fun y hy => hok y (by simp [hy])
    rw [G_cons] at hc
    by_cases hi : inl B x = true
    · have hrl := record_len B x h9 hi
      by_cases hfit : b.length + (record x).length ≤ B
      · rw [write_fit B b x hi hfit] at hc; simp at hc
        exact ih _ (by simp; omega) hxs c hc
      · have hbne : b ≠ [] := by intro h; subst h; simp at hfit; omega
        rw [write_spill B b x hi (by omega) hbne] at hc; simp at hc
        rcases hc with rfl | hc
        · exact ⟨hbne, by omega⟩
        · exact ih _ hrl hxs c hc
    · have hi' : inl B x = false := by simpa using hi
      have hpl : (payload x).length < 256^8 := by
        cases x <;> simp_all [payload, W.ok, inl]
      have hpn := blob_payload_ne B x h9 hi'
      by_cases hfit : b.length + 1 ≤ B
      · rw [write_blob_fit B b x hi' hfit] at hc; simp at hc
        rcases hc with rfl | rfl | hc
        · simp; omega
        · exact ⟨hpn, hpl⟩
        · exact ih _ (by simp) hxs c hc
      · have hbne : b ≠ [] := by intro h; subst h; simp at hfit; omega
        rw [write_blob_spill B b x hi' (by omega) hbne] at hc; simp at hc
        rcases hc with rfl | rfl | rfl | hc
        · exact ⟨hbne, by omega⟩
        · simp [TYPE_BLOB]
        · exact ⟨hpn, hpl⟩
        · exact ih _ (by simp) hxs c hc

theorem writer_layout (xs : List W) : ∀ (b : Bytes) (t : σ), (∀ x ∈ xs, x.ok) → b.length ≤ B →
    (b = [] → readSeq S xs ⟨[], cat (G B b xs) t⟩ = .ok (xs, ⟨[], cat [] t⟩)) ∧
    (b ≠ [] → ∃ T Cs, G B b xs = (b ++ T) :: Cs ∧
        readSeq S xs ⟨T, cat Cs t⟩ = .ok (xs, ⟨[], cat [] t⟩)) := by
  induction xs with
  | nil =>
    intro b t _ _
    refine ⟨fun hb => ?_, fun hb => ?_⟩
    · subst hb; simp [G_nil, readSeq]
    · exact ⟨[], [], by simp [G_nil, hb], by simp [readSeq]⟩
  | cons x xs ih =>
    intro b t hok hbB
    have hx : x.ok := hok x (by simp)
    have hxs : ∀ y ∈ xs, y.ok := fun y hy => hok y (by simp [hy])
    have hbd := G_bound B h9 hB (x :: xs) b hbB hok
    by_cases hi : inl B x = true
    · have hrl := record_len B x h9 hi
      have hrne := rec_ne x
      -- the record lands at the head of the chunk `G B b' xs` starts with
      have land : ∀ b', b' ≠ [] → b'.length ≤ B →
          (∀ c ∈ G B b' xs, c ≠ [] ∧ c.length < 256^8) →
          ∃ T' Cs', G B b' xs = (b' ++ T') :: Cs' ∧
            readSeq S xs ⟨T', cat Cs' t⟩ = .ok (xs, ⟨[], cat [] t⟩) :=
        fun b' h1 h2 _ => (ih b' t hxs h2).2 h1
      by_cases hfit : b.length + (record x).length ≤ B
      · have hw := write_fit B b x hi hfit
        rw [G_cons, hw] at hbd ⊢; simp only [List.nil_append] at hbd ⊢
        obtain ⟨T', Cs', hG, hR⟩ := land (b ++ record x) (by simp [hrne]) (by simp; omega) hbd
        rw [hG] at hbd ⊢
        have hh := hbd _ (List.mem_cons_self ..)
        refine ⟨fun hb => ?_, fun _ => ⟨record x ++ T', Cs', by simp, ?_⟩⟩
        · subst hb; simp only [List.nil_append] at hh ⊢
          exact readSeq_cons_ok S x xs _ _ _ x xs
            (readW_inline S x _ T' _ hx (load_head B S cat hload hblob h9 hB _ _ _ hh.1 hh.2)) hR
        · obtain ⟨c, cs, hc⟩ : ∃ c cs, record x = c :: cs := by
            cases h : record x with
            | nil => exact absurd h hrne
            | cons c cs => exact ⟨c, cs, rfl⟩
          exact readSeq_cons_ok S x xs _ _ _ x xs
            (readW_inline S x _ T' _ hx (by rw [hc]; simp [ensure])) hR
      · have hbne : b ≠ [] := by intro h; subst h; simp at hfit; omega
        have hw := write_spill B b x hi (by omega) hbne
        rw [G_cons, hw] at hbd ⊢
        have hbd' : ∀ c ∈ G B (record x) xs, c ≠ [] ∧ c.length < 256^8 :=
          fun c hc => hbd c (by simp [hc])
        obtain ⟨T', Cs', hG, hR⟩ := land (record x) hrne hrl hbd'
        refine ⟨fun hb => absurd hb hbne, fun _ => ⟨[], G B (record x) xs, by simp, ?_⟩⟩
        rw [hG] at hbd' ⊢
        have hh := hbd' _ (List.mem_cons_self ..)
        exact readSeq_cons_ok S x xs _ _ _ x xs
          (readW_inline S x _ T' _ hx (load_head B S cat hload hblob h9 hB _ _ _ hh.1 hh.2)) hR
    · have hi' : inl B x = false := by simpa using hi
      have hpn := blob_payload_ne B x h9 hi'
      have hpl : (payload x).length < 256^8 := by
        cases x <;> simp_all [payload, W.ok, inl]
      have hxb : ∃ d0, x = .buf d0 := by cases x <;> simp_all [inl]
      have hpx : x = .buf (payload x) := by obtain ⟨d0, rfl⟩ := hxb; rfl
      have hR := (ih [] (t := t) hxs (by simp)).1 rfl
      have rd : ∀ s, ensure S s = .ok ⟨[TYPE_BLOB], cat (payload x :: G B [] xs) t⟩ →
          readSeq S (x :: xs) s = .ok (x :: xs, ⟨[], cat [] t⟩) := by
        intro s hs
        have := readW_blob S x s _ (payload x) (cat (G B [] xs) t) hxb hs
          (hblob _ _ _ hpn hpl)
        rw [← hpx] at this
        exact readSeq_cons_ok S x xs _ _ _ x xs this hR
      by_cases hfit : b.length + 1 ≤ B
      · have hw := write_blob_fit B b x hi' hfit
        rw [G_cons, hw]; simp only [List.cons_append, List.nil_append]
        refine ⟨fun hb => ?_, fun _ => ⟨[TYPE_BLOB], _, rfl, rd _ (by simp [ensure])⟩⟩
        subst hb; apply rd
        exact load_head B S cat hload hblob h9 hB _ _ _ (by simp) (by simp [TYPE_BLOB])
      · have hbne : b ≠ [] := by intro h; subst h; simp at hfit; omega
        have hw := write_blob_spill B b x hi' (by omega) hbne
        rw [G_cons, hw]; simp only [List.cons_append, List.nil_append]
        refine ⟨fun hb => absurd hb hbne, fun _ => ⟨[], [TYPE_BLOB] :: payload x :: G B [] xs, by simp, rd _ ?_⟩⟩
        exact load_head B S cat hload hblob h9 hB _ _ _ (by simp) (by simp [TYPE_BLOB])

/-- **Round trip.** Reading back the writer's output with the matching reader returns
exactly the values written, for every source satisfying the chunk contract. -/
theorem roundtrip (xs : List W) (t : σ) (hok : ∀ x ∈ xs, x.ok) :
    readSeq S xs ⟨[], cat (writeAll B xs) t⟩ = .ok (xs, ⟨[], cat [] t⟩) :=
  (writer_layout B S cat hload hblob h9 hB xs [] t hok (by simp)).1 rfl

end

/-- RDB instance: `BufferedWriter` → Redis string buffers → `BufferedReader::new`. -/
theorem rdb_roundtrip (B : Nat) (h9 : 9 ≤ B) (hB : B < 256^8) (xs : List W)
    (hok : ∀ x ∈ xs, x.ok) (rest : List Bytes) :
    readSeq (listSrc false) xs ⟨[], writeAll B xs ++ rest⟩ = .ok (xs, ⟨[], rest⟩) := by
  have := roundtrip B (listSrc false) (fun cs t => cs ++ t)
    (by intro c cs t _ _; simp [listSrc]) (by intro c cs t _ _; simp [listSrc]) h9 hB xs rest hok
  simpa using this

/-- Production buffer size. -/
theorem rdb_roundtrip_256k (xs : List W) (hok : ∀ x ∈ xs, x.ok) :
    readSeq (listSrc false) xs ⟨[], writeAll 256000 xs⟩ = .ok (xs, ⟨[], []⟩) := by
  simpa using rdb_roundtrip 256000 (by omega) (by simp) xs hok []

/-- Pipe instance (`GRAPH.COPY`): `PipeWriter` → fd → `PipeReader`; what is left on the
pipe afterwards is the zero-length terminator, on which `load_chunk` reports end-of-stream
(`pipeLoad_end`). -/
theorem pipe_roundtrip (B : Nat) (h9 : 9 ≤ B) (hB : B < 256^8) (xs : List W)
    (hok : ∀ x ∈ xs, x.ok) :
    readSeq pipeSrc xs ⟨[], frame (writeAll B xs) ++ le 8 0⟩
      = .ok (xs, ⟨[], frame [] ++ le 8 0⟩) :=
  roundtrip B pipeSrc (fun cs t => frame cs ++ t)
    (by intro c cs t hc hl; simp only [pipeSrc]; exact pipeLoad_frame c cs t hc hl)
    (by intro c cs t hc hl; simp only [pipeSrc]; exact pipeLoad_frame c cs t hc hl)
    h9 hB xs (le 8 0) hok

/-! ## `VecWriter` / `BufferedReader::from_slice` (`GRAPH.RESTORE` replication) -/

/-- `VecWriter` (`:576-620`): every write inlined, no chunking, no blob sentinel. -/
def vecWrite (xs : List W) : Bytes := (xs.map record).flatten

/-- `VecWriter::into_vec` returns exactly the buffer: the concatenation of the records. -/
theorem vecWrite_cons (x : W) (xs : List W) : vecWrite (x :: xs) = record x ++ vecWrite xs := by
  simp [vecWrite]

theorem vec_roundtrip (xs : List W) (hok : ∀ x ∈ xs, x.ok) (T : Bytes) :
    readSeq (listSrc true) xs ⟨vecWrite xs ++ T, []⟩ = .ok (xs, ⟨T, []⟩) := by
  induction xs with
  | nil => simp [vecWrite, readSeq]
  | cons x xs ih =>
    have hx : x.ok := hok x (by simp)
    have hrne := rec_ne x
    obtain ⟨c, cs, hc⟩ : ∃ c cs, record x = c :: cs := by
      cases h : record x with
      | nil => exact absurd h hrne
      | cons c cs => exact ⟨c, cs, rfl⟩
    apply readSeq_cons_ok _ x xs _ ⟨vecWrite xs ++ T, []⟩ _ x xs
    · apply readW_inline _ x _ _ _ hx
      rw [vecWrite_cons, List.append_assoc, hc]; simp [ensure]
    · exact ih (fun y hy => hok y (by simp [hy]))

/-- …but one read past the end of a `from_slice` reader is not an error: `ensure_available`
calls `load_chunk`, i.e. `RedisModule_LoadStringBuffer(NULL)` (`:213`, `rdb` is NULL from
`:205`). Every truncated `GRAPH.RESTORE` payload therefore reaches it — the root cause of
#2537 (`GRAPH.RESTORE k ""` SIGSEGV), and not only for the empty payload. -/
theorem from_slice_past_end (x : W) :
    readW (listSrc true) x ⟨[], []⟩ = .crash := by
  cases x <;> simp [readW, readTag, ensure, listSrc, Res.bind]

end RedisLayer.BufferedIO
