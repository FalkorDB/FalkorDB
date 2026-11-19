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
    (by intro c cs t hc _; simp [listSrc, hc]) (by intro c cs t _ _; simp [listSrc]) h9 hB xs rest hok
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

/-- One read past the end of a `from_slice` reader is an error (#3643, `load_chunk` `:220`
refuses the NULL handle). Before #3643 it called `RedisModule_LoadStringBuffer(NULL)`, so
every truncated `GRAPH.RESTORE` payload crashed the server (#2537). -/
theorem from_slice_past_end_err (x : W) (l : List Bytes) :
    readW (listSrc true) x ⟨[], l⟩ = .err := by
  cases x <;> simp [readW, readTag, ensure, listSrc, Res.bind]

/-- An empty RDB chunk is refused (#3643) instead of indexing `buf[0]` of it. -/
theorem rdb_empty_chunk_err (x : W) (cs : List Bytes) :
    readW (listSrc false) x ⟨[], [] :: cs⟩ = .err := by
  cases x <;> simp [readW, readTag, ensure, listSrc, Res.bind]

/-! ## No read crashes

`Res.crash` is only reachable by indexing an empty chunk. A source whose `load` never crashes
and never returns an empty chunk, and whose `blob` never crashes, makes every read total. -/

section NoCrash
variable {σ : Type} (S : Src σ)

theorem bind_ne_crash {α β} (m : Res α) (f : α → Res β) (hm : m ≠ .crash)
    (hf : ∀ a, f a ≠ .crash) : m.bind f ≠ .crash := by
  cases m <;> simp_all [Res.bind]

theorem readBytes_ne_crash (n : Nat) (r : Rd σ) : readBytes n r ≠ .crash := by
  unfold readBytes; split <;> simp

theorem ensure_ok_or_err
    (hl : ∀ s, S.load s ≠ .crash) (hne : ∀ s c s', S.load s = .ok (c, s') → c ≠ [])
    (r : Rd σ) : ensure S r = .err ∨ ∃ r', ensure S r = .ok r' ∧ r'.rest ≠ [] := by
  unfold ensure
  split
  · cases h : S.load r.src with
    | ok p => exact Or.inr ⟨⟨p.1, p.2⟩, by simp [Res.bind], hne _ _ _ h⟩
    | err => exact Or.inl (by simp [Res.bind])
    | crash => exact absurd h (hl _)
  · exact Or.inr ⟨r, rfl, by assumption⟩

theorem ensure_bind_ne_crash {β} (hl : ∀ s, S.load s ≠ .crash)
    (hne : ∀ s c s', S.load s = .ok (c, s') → c ≠ []) (r : Rd σ) (f : Rd σ → Res β)
    (hf : ∀ r' : Rd σ, r'.rest ≠ [] → f r' ≠ .crash) : (ensure S r).bind f ≠ .crash := by
  rcases ensure_ok_or_err S hl hne r with h | ⟨r', h, hr⟩
  · simp [h, Res.bind]
  · simp only [h, Res.bind]; exact hf r' hr

theorem readTag_ne_crash (hl : ∀ s, S.load s ≠ .crash)
    (hne : ∀ s c s', S.load s = .ok (c, s') → c ≠ []) (e : Nat) (r : Rd σ) :
    readTag S e r ≠ .crash := by
  unfold readTag
  apply ensure_bind_ne_crash S hl hne
  intro ⟨rest, s⟩ hr
  cases rest with
  | nil => exact absurd rfl hr
  | cons t rs => simp only; split <;> simp

/-- **No read crashes** (#3643): with such a source, every read returns a value or an error. -/
theorem readW_no_crash (hl : ∀ s, S.load s ≠ .crash)
    (hne : ∀ s c s', S.load s = .ok (c, s') → c ≠ []) (hb : ∀ s, S.blob s ≠ .crash)
    (x : W) (r : Rd σ) : readW S x r ≠ .crash := by
  have hT := readTag_ne_crash S hl hne
  have hB := readBytes_ne_crash (σ := σ)
  cases x with
  | buf _ =>
    simp only [readW]
    apply ensure_bind_ne_crash S hl hne
    intro ⟨rest, s⟩ hr
    cases rest with
    | nil => exact absurd rfl hr
    | cons t rs =>
      simp only
      split
      · exact bind_ne_crash _ _ (hB _ _) fun _ =>
          bind_ne_crash _ _ (hB _ _) fun _ => by simp
      · split
        · exact bind_ne_crash _ _ (hb _) fun _ => by simp
        · simp
  | _ =>
    simp only [readW]
    exact bind_ne_crash _ _ (hT _ _) fun _ =>
      bind_ne_crash _ _ (hB _ _) fun _ => by simp

theorem readSeq_no_crash (hl : ∀ s, S.load s ≠ .crash)
    (hne : ∀ s c s', S.load s = .ok (c, s') → c ≠ []) (hb : ∀ s, S.blob s ≠ .crash)
    (xs : List W) (r : Rd σ) : readSeq S xs r ≠ .crash := by
  induction xs generalizing r with
  | nil => simp [readSeq]
  | cons x xs ih =>
    simp only [readSeq]
    exact bind_ne_crash _ _ (readW_no_crash S hl hne hb x r) fun _ =>
      bind_ne_crash _ _ (ih _) fun _ => by simp

end NoCrash

/-- `BufferedReader` (`new` and `from_slice`) never crashes on any input (#3643). -/
theorem listSrc_no_crash (null : Bool) (xs : List W) (r : Rd (List Bytes)) :
    readSeq (listSrc null) xs r ≠ .crash := by
  apply readSeq_no_crash
  · intro l; cases null <;> cases l <;> simp [listSrc] <;> split <;> simp
  · intro l c l' h
    cases null <;> cases l <;> simp [listSrc] at h
    split at h <;> simp_all
  · intro l; cases null <;> cases l <;> simp [listSrc]

theorem pipeLoad_ne (s c s' : Bytes) (h : pipeLoad s = .ok (c, s')) : c ≠ [] := by
  unfold pipeLoad at h
  dsimp only at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · split at h
      · simp at h
      · simp at h
        rw [← h.1]; intro he
        have := congrArg List.length he
        rw [List.length_take, List.length_nil] at this; omega

theorem pipeLoad_ne_crash (s : Bytes) : pipeLoad s ≠ .crash := by
  unfold pipeLoad; dsimp only; split
  · simp
  · split
    · simp
    · split <;> simp

/-- `PipeReader` never crashes either (`load_chunk` refuses a zero length). -/
theorem pipeSrc_no_crash (xs : List W) (r : Rd Bytes) : readSeq pipeSrc xs r ≠ .crash :=
  readSeq_no_crash pipeSrc pipeLoad_ne_crash pipeLoad_ne pipeLoad_ne_crash xs r

/-! ## `Reader::remaining` (`:193`, #3643) and `check_count_fits`

A `from_slice` reader holds the whole payload: `remaining` is `buf.len() - pos`, i.e. the
length of `rest`; a stream reader returns `None`. Every `u`/`s`/`d`/`buf` record a
`from_slice` reader accepts takes at least 9 bytes, so `check_count_fits(r, _, n, 9)` refuses
only counts whose `n` records could not all be read: `remaining_bounds_reads`. -/

def remaining (null : Bool) (r : Rd (List Bytes)) : Option Nat :=
  if null then some r.rest.length else none

/-- A record of at least 9 bytes: everything but `write_float`. -/
def W.nine : W → Prop
  | .f _ => False
  | _ => True

theorem readBytes_len {σ} (n : Nat) (r r' : Rd σ) (b : Bytes) (h : readBytes n r = .ok (b, r')) :
    r.rest.length = r'.rest.length + n ∧ r'.src = r.src := by
  unfold readBytes at h; split at h
  · simp at h; rw [← h.2]; simp; omega
  · simp at h

theorem readTag_slice_len (e : Nat) (r r' : Rd (List Bytes))
    (h : readTag (listSrc true) e r = .ok r') :
    r.rest.length = r'.rest.length + 1 ∧ r'.src = r.src := by
  obtain ⟨rest, s⟩ := r
  cases rest with
  | nil => simp [readTag, ensure, listSrc, Res.bind] at h
  | cons t rs =>
    simp only [readTag, ensure, reduceCtorEq, ite_false, Res.bind] at h
    split at h
    · simp at h; rw [← h]; simp
    · simp at h

theorem readW_slice_len (x : W) (hx : x.nine) (r r' : Rd (List Bytes)) (y : W)
    (h : readW (listSrc true) x r = .ok (y, r')) :
    r'.rest.length + 9 ≤ r.rest.length ∧ r'.src = r.src := by
  have tagged : ∀ (e : Nat) (g : Bytes → W),
      ((readTag (listSrc true) e r).bind fun r =>
        (readBytes 8 r).bind fun (b, r) => Res.ok (g b, r)) = .ok (y, r') →
      r'.rest.length + 9 ≤ r.rest.length ∧ r'.src = r.src := by
    intro e g h
    cases h1 : readTag (listSrc true) e r with
    | ok r1 =>
      rw [h1] at h; simp only [Res.bind] at h
      cases h2 : readBytes 8 r1 with
      | ok p =>
        rw [h2] at h; simp only [Res.bind] at h
        obtain ⟨b, r2⟩ := p
        simp at h; rw [← h.2]
        have a1 := readTag_slice_len e r r1 h1
        have a2 := readBytes_len 8 r1 r2 b h2
        exact ⟨by omega, by rw [a2.2, a1.2]⟩
      | err => rw [h2] at h; simp [Res.bind] at h
      | crash => rw [h2] at h; simp [Res.bind] at h
    | err => rw [h1] at h; simp [Res.bind] at h
    | crash => rw [h1] at h; simp [Res.bind] at h
  cases x with
  | u _ => exact tagged TYPE_UNSIGNED (fun b => .u (fromLE b)) h
  | s _ => exact tagged TYPE_SIGNED (fun b => .s (ofU (fromLE b))) h
  | d _ => exact tagged TYPE_DOUBLE (fun b => .d (fromLE b)) h
  | f _ => exact absurd hx (by simp [W.nine])
  | buf _ =>
    obtain ⟨rest, s⟩ := r
    cases rest with
    | nil => simp [readW, ensure, listSrc, Res.bind] at h
    | cons t rs =>
      simp only [readW, ensure, reduceCtorEq, ite_false, Res.bind] at h
      split at h
      · cases h2 : readBytes 8 ⟨rs, s⟩ with
        | ok p =>
          rw [h2] at h; simp only [Res.bind] at h
          obtain ⟨lb, r2⟩ := p
          have a2 := readBytes_len 8 ⟨rs, s⟩ r2 lb h2
          cases h3 : readBytes (fromLE lb) r2 with
          | ok q =>
            rw [h3] at h; simp only [Res.bind] at h
            obtain ⟨ds, r3⟩ := q
            have a3 := readBytes_len _ r2 r3 ds h3
            simp at h; rw [← h.2]
            simp at a2 ⊢
            exact ⟨by omega, by rw [a3.2, a2.2]⟩
          | err => rw [h3] at h; simp at h
          | crash => rw [h3] at h; simp at h
        | err => rw [h2] at h; simp at h
        | crash => rw [h2] at h; simp at h
      · split at h
        · simp [listSrc] at h
        · simp at h

/-- **`check_count_fits` is sound** on a `from_slice` reader: if `n` records of at least
9 bytes each can be read, then `n * 9 ≤ remaining`. So the new refusal (`Header::decode`
`:196`, `relationship type`, `min_bytes` 9) only turns an `Err` the loop would have
reached anyway into an earlier one. -/
theorem remaining_bounds_reads (xs : List W) (hx : ∀ x ∈ xs, x.nine)
    (r r' : Rd (List Bytes)) (ys : List W)
    (h : readSeq (listSrc true) xs r = .ok (ys, r')) :
    ∃ k, remaining true r = some k ∧ xs.length * 9 ≤ k := by
  refine ⟨r.rest.length, by simp [remaining], ?_⟩
  induction xs generalizing r ys with
  | nil => simp
  | cons x xs ih =>
    simp only [readSeq] at h
    cases h1 : readW (listSrc true) x r with
    | ok p =>
      rw [h1] at h; simp only [Res.bind] at h
      obtain ⟨y, r1⟩ := p
      have a1 := readW_slice_len x (hx x (by simp)) r r1 y h1
      cases h2 : readSeq (listSrc true) xs r1 with
      | ok q =>
        obtain ⟨ys1, r2⟩ := q
        rw [h2] at h; simp only [Res.bind] at h
        injection h with h; injection h with _ h; subst h
        have := ih (fun z hz => hx z (by simp [hz])) r1 ys1 h2
        simp; omega
      | err => rw [h2] at h; simp [Res.bind] at h
      | crash => rw [h2] at h; simp [Res.bind] at h
    | err => rw [h1] at h; simp [Res.bind] at h
    | crash => rw [h1] at h; simp [Res.bind] at h

/-- A stream reader reports nothing: `check_count_fits` is a no-op there. -/
theorem remaining_stream (r : Rd (List Bytes)) : remaining false r = none := rfl

end RedisLayer.BufferedIO
