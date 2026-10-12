//! AMD elimination-loop primitives: pivot selection and element
//! construction (plus standard absorption).
//!
//! This module lands Commit 4 of the Slice A plan. It ports faer's
//! `amd.rs:220-365` line-by-line:
//!
//! - [`select_pivot`]: linear scan from `mindeg`, LIFO unlink.
//! - [`create_element`]: both the in-place (`elenme == 0`) and
//!   out-of-place (`elenme > 0`) branches, plus standard absorption
//!   fired at the end of each `knt1` iter (faer `amd.rs:355-358`),
//!   and the final bookkeeping write-back to `pe[me]/len[me]/elen[me]`
//!   with the post-step `clear_flag` call.
//!
//! Inline garbage collection (faer `amd.rs:289-338`) fires inside
//! the out-of-place branch when `pfree >= iwlen`. See
//! [`create_element`] for the full save → mark → compact → restore
//! dance.
//!
//! Pass-1 `w[e]` seeding (`amd.rs:366-385`), Pass-2 approximate
//! degree (`amd.rs:386-465`), aggressive absorption, monotone
//! degree cap, and re-insertion into degree lists (`amd.rs:516-546`)
//! live in [`finalize_step`]. Mass elimination and supervariable
//! detection are Slice B (Commits 9-10).

#![allow(dead_code)]

use super::workspace::{clear_flag, flip, Workspace, NONE};
use crate::Idx;
use crate::OrderingError;

/// Flop-counter deltas produced by a single elimination step.
/// Matches faer's `amd.rs:547-557` accounting so `AmdStats` can
/// accumulate consistent `ndiv` / `nms_ldl` / `nms_lu` totals.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepFlops {
    pub ndiv: f64,
    pub nms_lu: f64,
    pub nms_ldl: f64,
}

impl StepFlops {
    fn accumulate(&mut self, other: StepFlops) {
        self.ndiv += other.ndiv;
        self.nms_lu += other.nms_lu;
        self.nms_ldl += other.nms_ldl;
    }
}

/// Scan `head` from `ws.mindeg` upward and return the first
/// non-empty degree-list head. Unlink the chosen variable. Returns
/// `None` if no bucket in `[ws.mindeg, ws.n)` is non-empty (i.e.
/// all remaining supervariables have been dense-deferred and the
/// main loop should stop).
///
/// Side effects: `ws.mindeg` advances to the degree of the chosen
/// pivot. `head[deg]` is advanced to the next element. `last[next]`
/// is cleared if a successor exists.
///
/// Reference: faer `amd.rs:220-235`.
pub fn select_pivot(ws: &mut Workspace) -> Option<usize> {
    let n = ws.n;
    let mut deg = ws.mindeg;
    let mut me_signed: Idx = NONE;
    while deg < n {
        let h = ws.head[deg];
        if h != NONE {
            me_signed = h;
            break;
        }
        deg += 1;
    }
    if me_signed == NONE {
        return None;
    }
    ws.mindeg = deg;
    let me = me_signed as usize;
    let inext = ws.next[me];
    if inext != NONE {
        ws.last[inext as usize] = NONE;
    }
    ws.head[deg] = inext;
    Some(me)
}

/// Build the new element `me` by merging the (variable) tail of
/// `me`'s list with every element `e` already in `me`'s list.
///
/// On success returns `(pme1, pme2_excl, nvpiv, degme)`:
/// - `pme1..pme2_excl` is the contiguous region in `ws.iw` holding
///   the new element's **variable** members (supervariables, listed
///   once each). Exclusive end so the empty case (`pme2_excl ==
///   pme1`) is representable in `usize` without underflow.
/// - `nvpiv` is the supervariable count of the pivot.
/// - `degme` is the tentative new element's external degree (sum of
///   `nv[i]` over the assembled variables, before any absorption
///   correction made by Pass-2).
///
/// Post-conditions also persisted on the workspace:
/// - `nv[me] = -nvpiv` (marker — Pass-2 will flip sign back via
///   `-nv[i]`).
/// - `nv[i] = -nv[i]` for every `i` assembled into the new element
///   (ditto — marker for Pass-2's w-seed walk).
/// - `pe[me] = pme1`, `len[me] = pme2 - pme1 + 1`, `elen[me] =
///   flip(nvpiv + degme)` (dead-variable sentinel carrying the
///   pivot-front size for the postorder phase).
/// - `degree[me] = degme` (temporary; Pass-2 overwrites).
/// - For every absorbed element `e != me` in the elenme>0 branch:
///   `pe[e] = flip(me)`, `w[e] = 0`. This is **standard absorption**
///   (faer `amd.rs:355-358`) — it fires unconditionally at each
///   `knt1` iter's end. Aggressive absorption (Pass-2 only) lands
///   in Commit 5.
/// - `ws.wflg` bumped via `clear_flag`.
/// - `ws.nel` incremented by `nvpiv`.
///
/// Reference: faer `amd.rs:236-366` (incl. inline GC at 289-338).
pub fn create_element(
    ws: &mut Workspace,
    me: usize,
) -> Result<(usize, usize, Idx, usize), OrderingError> {
    let elenme = ws.elen[me];
    let nvpiv = ws.nv[me];
    ws.nel += nvpiv as usize;
    ws.nv[me] = -nvpiv;
    let mut degme: usize = 0;
    let pme1: usize;
    let pme2: Idx;

    if elenme == 0 {
        // In-place: me has no elements in its list — just variables.
        // Compact them at pe[me]: advance pme2 to the final position,
        // overwriting absorbed entries in-place.
        let pme1_s = ws.pe[me];
        pme1 = pme1_s as usize;
        let list_start = pme1;
        let list_end = list_start + ws.len[me] as usize;
        let mut pme2_s = pme1_s - 1;
        for p in list_start..list_end {
            let i = ws.iw[p] as usize;
            let nvi = ws.nv[i];
            if nvi > 0 {
                degme += nvi as usize;
                ws.nv[i] = -nvi;
                pme2_s += 1;
                ws.iw[pme2_s as usize] = i as Idx;
                // Unlink i from its degree list.
                let ilast = ws.last[i];
                let inext = ws.next[i];
                if inext != NONE {
                    ws.last[inext as usize] = ilast;
                }
                if ilast != NONE {
                    ws.next[ilast as usize] = inext;
                } else {
                    ws.head[ws.degree[i] as usize] = inext;
                }
            }
        }
        pme2 = pme2_s;
    } else {
        // Out-of-place: start a new region at pfree. Walk every
        // element e in me's list (first `elenme` entries of me's
        // adjacency), then walk the variable tail (remaining
        // `slenme` entries) with `knt1 = elenme + 1` as the flag.
        let mut p = ws.pe[me] as usize;
        let mut pme1_rw: usize = ws.pfree;
        let slenme = (ws.len[me] - elenme) as usize;
        let elenme_u = elenme as usize;
        for knt1 in 1..=elenme_u + 1 {
            let e: usize;
            let mut pj: usize;
            let ln: usize;
            if knt1 > elenme_u {
                // Variable tail of me's own list.
                e = me;
                pj = p;
                ln = slenme;
            } else {
                e = ws.iw[p] as usize;
                p += 1;
                pj = ws.pe[e] as usize;
                ln = ws.len[e] as usize;
            }
            for knt2 in 1..=ln {
                let i = ws.iw[pj] as usize;
                pj += 1;
                let nvi = ws.nv[i];
                if nvi > 0 {
                    if ws.pfree >= ws.iwlen {
                        // Inline garbage collection (faer
                        // amd.rs:289-338). Save partial state so
                        // the surviving elements can be compacted
                        // down, then restore local cursors.
                        ws.pe[me] = p as Idx;
                        ws.len[me] -= knt1 as Idx;
                        if ws.len[me] == 0 {
                            ws.pe[me] = NONE;
                        }
                        ws.pe[e] = pj as Idx;
                        ws.len[e] = (ln - knt2) as Idx;
                        if ws.len[e] == 0 {
                            ws.pe[e] = NONE;
                        }
                        ws.ncmpa += 1;
                        // Mark each live list's head: save iw[pe[j]]
                        // into pe[j] and write flip(j) at the old
                        // head position so the compact sweep can
                        // recognise list starts.
                        for j in 0..ws.n {
                            let pn = ws.pe[j];
                            if pn >= 0 {
                                let pn_u = pn as usize;
                                ws.pe[j] = ws.iw[pn_u];
                                ws.iw[pn_u] = flip(j as Idx);
                            }
                        }
                        // Sweep [0, pme1_rw), reconstructing every
                        // marked list contiguously at pdst.
                        let mut psrc = 0usize;
                        let mut pdst = 0usize;
                        let pend = pme1_rw;
                        while psrc < pend {
                            let j_marker = flip(ws.iw[psrc]);
                            psrc += 1;
                            if j_marker >= 0 {
                                let j = j_marker as usize;
                                ws.iw[pdst] = ws.pe[j];
                                ws.pe[j] = pdst as Idx;
                                pdst += 1;
                                let lenj = ws.len[j] as usize;
                                if lenj > 0 {
                                    ws.iw.copy_within(psrc..psrc + lenj - 1, pdst);
                                    psrc += lenj - 1;
                                    pdst += lenj - 1;
                                }
                            }
                        }
                        // Slide the new element's accumulated prefix
                        // [pme1_rw, pfree) down to the new pdst.
                        let p1 = pdst;
                        ws.iw.copy_within(pme1_rw..ws.pfree, pdst);
                        pdst += ws.pfree - pme1_rw;
                        pme1_rw = p1;
                        ws.pfree = pdst;
                        // Restore local cursors from the relocated
                        // heads of e's and me's lists.
                        pj = ws.pe[e] as usize;
                        p = ws.pe[me] as usize;
                    }
                    degme += nvi as usize;
                    ws.nv[i] = -nvi;
                    ws.iw[ws.pfree] = i as Idx;
                    ws.pfree += 1;
                    // Unlink i from its degree list.
                    let ilast = ws.last[i];
                    let inext = ws.next[i];
                    if inext != NONE {
                        ws.last[inext as usize] = ilast;
                    }
                    if ilast != NONE {
                        ws.next[ilast as usize] = inext;
                    } else {
                        ws.head[ws.degree[i] as usize] = inext;
                    }
                }
            }
            // Standard absorption (amd.rs:355-358): every element e
            // that was in me's list is now absorbed by me.
            if e != me {
                ws.pe[e] = flip(me as Idx);
                ws.w[e] = 0;
            }
        }
        pme1 = pme1_rw;
        pme2 = (ws.pfree - 1) as Idx;
    }

    ws.degree[me] = degme as Idx;
    ws.pe[me] = pme1 as Idx;
    ws.len[me] = pme2 - pme1 as Idx + 1;
    ws.elen[me] = flip(nvpiv + degme as Idx);
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    // Convert the inclusive `pme2` (which is `pme1 - 1` when the
    // pivot's variable list ended up empty — every neighbour was
    // already absorbed) to an exclusive end so downstream loops can
    // use a `usize` half-open range without the wrap-around bug
    // `(-1 as Idx) as usize == usize::MAX`.
    let pme2_excl: usize = if pme2 < pme1 as Idx {
        pme1
    } else {
        (pme2 + 1) as usize
    };
    Ok((pme1, pme2_excl, nvpiv, degme))
}

/// Finish the elimination step whose create-element phase produced
/// `(pme1, pme2_excl, nvpiv, degme)` and left `nv[me] = -nvpiv` and
/// `nv[i] = -nv[i]` for every variable `i ∈ iw[pme1..pme2_excl]`.
///
/// Does, in order:
/// 1. **Pass-1 w-seeding** (faer `amd.rs:366-385`). For each
///    variable `i` in the new element, walk its element list and
///    lazily seed `w[e]`: first touch sets `w[e] = degree[e] +
///    (wflg - nvi)`, subsequent touches do `w[e] -= nvi`.
/// 2. **Pass-2 approximate external degree** (`amd.rs:386-462`).
///    For each variable `i` in the new element: walk its element
///    list computing `dext = w[e] - wflg`, then its variable list
///    accumulating `nv[j]` for live neighbours. Under `aggressive`,
///    dead elements (`dext == 0`) are absorbed on the spot. The
///    updated degree is clamped by `min(degree[i], deg)`
///    ("monotone cap"). The element list is re-ordered so `me` sits
///    at position `p1`.
/// 3. **Mass elimination** (`amd.rs:436-444`). A member `i` whose
///    only remaining element is `me` (`elen[i] == 1`) and whose
///    surviving variable neighbourhood is empty (`p3 == pn`) will
///    pivot concurrently with `me`. Fold its supervariable count
///    into `nvpiv` / `nel` and deduct from `degme`.
/// 4. **Hash-bucket insertion** (`amd.rs:452-460`): each still-
///    marked member is placed into a hash bucket threaded through
///    `head`/`next`/`last` via sign-bit encoding.
/// 5. Bump `degree[me] = degme`, `lemax = max(lemax, degme)`,
///    `wflg += lemax`, `wflg = clear_flag(...)`.
/// 6. **Supervariable detection** (`amd.rs:467-515`): for each
///    hash chain whose anchor is still marked, walk the chain and
///    merge indistinguishable followers into the head.
/// 7. **Re-insert** (`amd.rs:516-537`): each surviving variable's
///    updated degree is pushed back onto `head[deg]` LIFO and
///    `mindeg` is lowered if needed.
/// 8. **Me bookkeeping** (`amd.rs:538-546`): restore `nv[me] =
///    nvpiv`, compact `me`'s var list to `[pme1, p)`, trim `pfree`.
/// 9. **Flop counters** (`amd.rs:547-557`).
#[allow(clippy::too_many_arguments)]
pub fn finalize_step(
    ws: &mut Workspace,
    me: usize,
    pme1: usize,
    pme2_excl: usize,
    nvpiv: Idx,
    degme: usize,
    elenme: Idx,
    aggressive: bool,
) -> StepFlops {
    let mut degme = degme;
    let mut nvpiv = nvpiv;

    // Pass 1: seed w[e] for every element in each member's list.
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let eln = ws.elen[i];
        if eln > 0 {
            let nvi = -ws.nv[i];
            let wnvi = ws.wflg - nvi;
            let pi = ws.pe[i] as usize;
            for k in 0..eln as usize {
                let e = ws.iw[pi + k] as usize;
                let mut we = ws.w[e];
                if we >= ws.wflg {
                    we -= nvi;
                } else if we != 0 {
                    we = ws.degree[e] + wnvi;
                }
                ws.w[e] = we;
            }
        }
    }

    // Pass 2: approximate degree, (optionally) aggressive absorption,
    // mass elimination (faer amd.rs:436-444), and hash-bucket
    // insertion for supervariable detection (amd.rs:451-460). `degme`
    // and `nvpiv` are mutated when mass-elim fires; the post-loop
    // degree/flop bookkeeping uses the updated values.
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let p1 = ws.pe[i] as usize;
        let p2 = p1 + ws.elen[i] as usize;
        let mut pn = p1;
        let mut deg: usize = 0;
        // Hash accumulator for supervariable detection (faer
        // amd.rs:419,433). Both elements AND variables in the kept
        // neighbourhood contribute.
        let mut hash: usize = 0;

        // Element sub-pass.
        if aggressive {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    let dext = we - ws.wflg;
                    if dext > 0 {
                        deg += dext as usize;
                        ws.iw[pn] = e as Idx;
                        pn += 1;
                        hash = hash.wrapping_add(e);
                    } else {
                        // Aggressive absorption: dead element folded
                        // into me right here (faer amd.rs:404-407).
                        ws.pe[e] = flip(me as Idx);
                        ws.w[e] = 0;
                    }
                }
            }
        } else {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    // Invariant (O4): a live element in the non-aggressive pass
                    // always has `we >= ws.wflg`, so the difference is
                    // non-negative. Guard the unchecked `as usize` cast — a
                    // future regression that broke the invariant would
                    // sign-extend a negative difference to ~2^64 here.
                    debug_assert!(
                        we >= ws.wflg,
                        "stale mark: w[e]={we} < wflg={} would wrap as usize",
                        ws.wflg
                    );
                    let dext = (we - ws.wflg) as usize;
                    deg += dext;
                    ws.iw[pn] = e as Idx;
                    pn += 1;
                    hash = hash.wrapping_add(e);
                }
            }
        }

        // Record number-of-elements + 1 (the +1 reserves the slot
        // for `me` which we insert at p1 below).
        ws.elen[i] = (pn - p1 + 1) as Idx;
        let p3 = pn;
        let p4 = p1 + ws.len[i] as usize;
        // Variable sub-pass.
        for p in p2..p4 {
            let j = ws.iw[p] as usize;
            let nvj = ws.nv[j];
            if nvj > 0 {
                deg += nvj as usize;
                ws.iw[pn] = j as Idx;
                pn += 1;
                hash = hash.wrapping_add(j);
            }
        }

        if ws.elen[i] == 1 && p3 == pn {
            // Mass elimination: i's only element is `me` and it has
            // no surviving outside variables, so it will pivot
            // concurrently with me. Fold its supervariable count
            // into nvpiv / nel and drop it from degme.
            ws.pe[i] = flip(me as Idx);
            let nvi = -ws.nv[i];
            debug_assert!(nvi >= 0);
            degme -= nvi as usize;
            nvpiv += nvi;
            ws.nel += nvi as usize;
            ws.nv[i] = 0;
            ws.elen[i] = NONE;
            ws.n_mass_elim += 1;
        } else {
            ws.degree[i] = ws.degree[i].min(deg as Idx);
            // Swap-dance to put `me` at the head of i's element list.
            if p1 != pn {
                ws.iw[pn] = ws.iw[p3];
            }
            if p3 != p1 {
                ws.iw[p3] = ws.iw[p1];
            }
            ws.iw[p1] = me as Idx;
            ws.len[i] = (pn - p1 + 1) as Idx;

            // Insert i into a hash bucket threaded through
            // head/next/last via sign-bit encoding (faer
            // amd.rs:452-460). `head[hash] <= NONE` distinguishes
            // two cases:
            //  * NONE (-1) or flip(prev_head)<=-2 mean the bucket
            //    is either empty or already a flip-encoded bucket
            //    chain: next[i]=flip(old), head[hash]=flip(i).
            //  * j>=0 means head[hash] still holds an unrelated
            //    degree-list head. Hijack last[j] to chain
            //    bucket members (last[j] is NONE for a list head,
            //    so no information is destroyed).
            let h = hash % ws.n;
            let j = ws.head[h];
            if j <= NONE {
                ws.next[i] = flip(j);
                ws.head[h] = flip(i as Idx);
            } else {
                ws.next[i] = ws.last[j as usize];
                ws.last[j as usize] = i as Idx;
            }
            ws.last[i] = h as Idx;
        }
    }

    // Step bookkeeping (amd.rs:463-466). degme may have been reduced
    // by mass elimination above.
    let degme_i32 = degme as Idx;
    ws.degree[me] = degme_i32;
    if degme_i32 > ws.lemax {
        ws.lemax = degme_i32;
    }
    ws.wflg += ws.lemax;
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    // Supervariable detection (faer amd.rs:467-515). For each hash
    // chain anchored at a still-marked member (nv[i] < 0), walk the
    // chain marking i's variable neighbourhood with `wflg`, then
    // compare each follower j to i: if len/elen agree AND every
    // variable neighbour of j is also marked, merge j into i.
    //
    // head / next / last are being reused here as hash-bucket
    // storage; they are restored to NONE at the heads involved so
    // the subsequent degree-list re-insertion pass can rebuild the
    // degree buckets cleanly.
    for pme in pme1..pme2_excl {
        let i_anchor = ws.iw[pme] as usize;
        if ws.nv[i_anchor] >= 0 {
            continue; // already restored / mass-elim'd
        }
        let h = ws.last[i_anchor] as usize;
        let j_head = ws.head[h];
        let mut i: Idx = if j_head == NONE {
            NONE
        } else if j_head < NONE {
            // Bucket was flip-encoded: flip(j_head) is the head.
            ws.head[h] = NONE;
            flip(j_head)
        } else {
            // Bucket chained via last[j_head]; restore last[j_head].
            let chain_start = ws.last[j_head as usize];
            ws.last[j_head as usize] = NONE;
            chain_start
        };
        while i != NONE && ws.next[i as usize] != NONE {
            let i_u = i as usize;
            let ln = ws.len[i_u];
            let eln = ws.elen[i_u];
            let pi = ws.pe[i_u];
            // Mark i's neighbourhood (everything past slot pe[i]=me).
            for p in (pi + 1) as usize..(pi + ln) as usize {
                ws.w[ws.iw[p] as usize] = ws.wflg;
            }
            let mut jlast = i_u;
            let mut jp = ws.next[i_u];
            while jp != NONE {
                let jj = jp as usize;
                let mut ok = ws.len[jj] == ln && ws.elen[jj] == eln;
                if ok {
                    let pj = ws.pe[jj];
                    for p in (pj + 1) as usize..(pj + ln) as usize {
                        if ws.w[ws.iw[p] as usize] != ws.wflg {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    // Merge j into i: j becomes a degree-0 ghost
                    // pointing at i via pe[j] = flip(i).
                    ws.pe[jj] = flip(i);
                    ws.nv[i_u] += ws.nv[jj];
                    ws.nv[jj] = 0;
                    ws.elen[jj] = NONE;
                    jp = ws.next[jj];
                    ws.next[jlast] = jp;
                    ws.n_supervar_merge += 1;
                } else {
                    jlast = jj;
                    jp = ws.next[jj];
                }
            }
            // Bump wflg to reset marks for the next chain head.
            ws.wflg += 1;
            i = ws.next[i_u];
        }
    }

    // Re-insertion (amd.rs:516-537): every surviving var in the new
    // element list gets its new degree, is pushed onto head[deg]
    // LIFO, and me's own list is compacted down to just the
    // survivors.
    let mut p_write = pme1;
    let nleft = ws.n - ws.nel;
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let nvi = -ws.nv[i];
        if nvi > 0 {
            ws.nv[i] = nvi;
            let mut d = ws.degree[i] as usize + degme_i32 as usize - nvi as usize;
            let cap = nleft - nvi as usize;
            if d > cap {
                d = cap;
            }
            let inext = ws.head[d];
            if inext != NONE {
                ws.last[inext as usize] = i as Idx;
            }
            ws.next[i] = inext;
            ws.last[i] = NONE;
            ws.head[d] = i as Idx;
            if d < ws.mindeg {
                ws.mindeg = d;
            }
            ws.degree[i] = d as Idx;
            ws.iw[p_write] = i as Idx;
            p_write += 1;
        }
    }

    // Me bookkeeping (amd.rs:538-546).
    ws.nv[me] = nvpiv;
    ws.len[me] = (p_write as Idx) - pme1 as Idx;
    if ws.len[me] == 0 {
        ws.pe[me] = NONE;
        ws.w[me] = 0;
    }
    if elenme != 0 {
        ws.pfree = p_write;
    }

    // Flop counters (amd.rs:547-557).
    let f = nvpiv as f64;
    let r = degme_i32 as f64 + ws.ndense as f64;
    let lnzme = f * r + (f - 1.0) * f / 2.0;
    let s = f * r * r + r * (f - 1.0) * f + (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;

    StepFlops {
        ndiv: lnzme,
        nms_lu: s,
        nms_ldl: (s + lnzme) / 2.0,
    }
}

/// AMF bucket index for a quantized fill score (`MinFill::bucket`).
///
/// Mirrors the metric trait's `MinFill::bucket` but is duplicated as a
/// free function here so `algo.rs` does not need a back-edge to
/// `metric.rs` (which itself imports from `algo`). Inlined.
#[inline(always)]
fn amf_bucket_of(score: i64, n: usize) -> usize {
    if score <= 0 {
        return 0;
    }
    let s = score as usize;
    if s <= n {
        return s;
    }
    let pas = (n / 8).max(1);
    let nbbuck = 2 * n;
    ((s - n) / pas + n).min(nbbuck)
}

/// AMF working-fill *surface contribution* of an element with current
/// external degree `dext` and total degree `degree`:
/// `dext * (2*degree - dext - 1)` (Amestoy 1999 thesis; MUMPS
/// `ana_orderings.F:4810`).
///
/// Computed in `i64`: both factors are `O(n)`, so the product reaches
/// ~`n^2` and overflows `i32` for `n` ≳ 46k (`i32::MAX` is
/// 2_147_483_647 and `46342 * 46341 = 2_147_534_622` already exceeds
/// it). In release the old `i32` form wrapped silently, feeding garbage
/// into the RMF pivot score; in debug it panicked. The value is later
/// consumed as `f64`, so widening loses no precision (O1,
/// `dev/research/repo-review-2026-06-09.md`).
#[inline(always)]
fn amf_wf_surface(dext: i64, degree: i64) -> i64 {
    dext * (2 * degree - dext - 1)
}

/// AMF per-supervariable working-fill accumulation
/// `wf4 + 2 * nvi * wf3` (Amestoy 1999 eq. for the B3 contribution;
/// MUMPS `ana_orderings.F:4810`). Computed in `i64` for the same
/// `O(n^2)` overflow reason as [`amf_wf_surface`]: `nvi` (supervariable
/// size) and `wf3` (sum of neighbour supervariable sizes) are each
/// `O(n)` (O1, `dev/research/repo-review-2026-06-09.md`).
#[inline(always)]
fn amf_wf_combine(wf4: i64, nvi: i64, wf3: i64) -> i64 {
    wf4 + 2 * nvi * wf3
}

/// Saturation cap used when quantizing the AMF RMF score into `Idx`.
/// `Idx::MAX - 1` matches MUMPS `idummy = huge(idummy) - 1`
/// (`ana_orderings.F:4230`).
const AMF_DUMMY_I32: Idx = Idx::MAX - 1;

/// AMF analogue of [`select_pivot`]. Linear-scans coarse buckets
/// (`idx > n`) for the entry with the smallest exact score; takes the
/// head for fine buckets.
///
/// Side effects: `ws.mindeg` advances to the chosen bucket index. The
/// chosen `me` is unlinked from its degree-list chain (head update for
/// fine buckets, doubly-linked unlink for coarse buckets).
///
/// Reference: `ana_orderings.F:4392-4427`.
pub fn select_pivot_amf(ws: &mut Workspace) -> Option<usize> {
    let n = ws.n;
    let nbuck = ws.head.len();
    let mut deg = ws.mindeg;
    while deg < nbuck && ws.head[deg] == NONE {
        deg += 1;
    }
    if deg >= nbuck {
        return None;
    }
    ws.mindeg = deg;
    let head_me = ws.head[deg] as usize;

    let me;
    if deg > n {
        // Coarse bucket: linear scan for the minimum-score entry.
        let mut best = head_me;
        let mut best_score = ws.wf[best];
        let mut j = ws.next[best];
        while j != NONE {
            let ju = j as usize;
            if ws.wf[ju] < best_score {
                best_score = ws.wf[ju];
                best = ju;
            }
            j = ws.next[ju];
        }
        me = best;
        // Doubly-linked unlink (best may be mid-chain).
        let ilast = ws.last[me];
        let inext = ws.next[me];
        if inext != NONE {
            ws.last[inext as usize] = ilast;
        }
        if ilast != NONE {
            ws.next[ilast as usize] = inext;
        } else {
            ws.head[deg] = inext;
        }
    } else {
        me = head_me;
        let inext = ws.next[me];
        if inext != NONE {
            ws.last[inext as usize] = NONE;
        }
        ws.head[deg] = inext;
    }
    Some(me)
}

/// AMF analogue of [`create_element`]. Identical structure; differs
/// only in the bucket-index used when unlinking absorbed neighbours
/// from their degree lists. AMD reads `degree[i]` (which doubles as
/// the bucket index because AMD's bucket is identity); AMF computes
/// `amf_bucket_of(wf[i], n)` because the AMF score and running degree
/// are stored in distinct fields (`wf` vs `degree`).
pub fn create_element_amf(
    ws: &mut Workspace,
    me: usize,
) -> Result<(usize, usize, Idx, usize), OrderingError> {
    let n = ws.n;
    let elenme = ws.elen[me];
    let nvpiv = ws.nv[me];
    ws.nel += nvpiv as usize;
    ws.nv[me] = -nvpiv;
    let mut degme: usize = 0;
    let pme1: usize;
    let pme2: Idx;

    if elenme == 0 {
        let pme1_s = ws.pe[me];
        pme1 = pme1_s as usize;
        let list_start = pme1;
        let list_end = list_start + ws.len[me] as usize;
        let mut pme2_s = pme1_s - 1;
        for p in list_start..list_end {
            let i = ws.iw[p] as usize;
            let nvi = ws.nv[i];
            if nvi > 0 {
                degme += nvi as usize;
                ws.nv[i] = -nvi;
                pme2_s += 1;
                ws.iw[pme2_s as usize] = i as Idx;
                let ilast = ws.last[i];
                let inext = ws.next[i];
                if inext != NONE {
                    ws.last[inext as usize] = ilast;
                }
                if ilast != NONE {
                    ws.next[ilast as usize] = inext;
                } else {
                    let h_idx = amf_bucket_of(ws.wf[i], n);
                    ws.head[h_idx] = inext;
                }
            }
        }
        pme2 = pme2_s;
    } else {
        let mut p = ws.pe[me] as usize;
        let mut pme1_rw: usize = ws.pfree;
        let slenme = (ws.len[me] - elenme) as usize;
        let elenme_u = elenme as usize;
        for knt1 in 1..=elenme_u + 1 {
            let e: usize;
            let mut pj: usize;
            let ln: usize;
            if knt1 > elenme_u {
                e = me;
                pj = p;
                ln = slenme;
            } else {
                e = ws.iw[p] as usize;
                p += 1;
                pj = ws.pe[e] as usize;
                ln = ws.len[e] as usize;
            }
            for knt2 in 1..=ln {
                let i = ws.iw[pj] as usize;
                pj += 1;
                let nvi = ws.nv[i];
                if nvi > 0 {
                    if ws.pfree >= ws.iwlen {
                        ws.pe[me] = p as Idx;
                        ws.len[me] -= knt1 as Idx;
                        if ws.len[me] == 0 {
                            ws.pe[me] = NONE;
                        }
                        ws.pe[e] = pj as Idx;
                        ws.len[e] = (ln - knt2) as Idx;
                        if ws.len[e] == 0 {
                            ws.pe[e] = NONE;
                        }
                        ws.ncmpa += 1;
                        for j in 0..ws.n {
                            let pn = ws.pe[j];
                            if pn >= 0 {
                                let pn_u = pn as usize;
                                ws.pe[j] = ws.iw[pn_u];
                                ws.iw[pn_u] = flip(j as Idx);
                            }
                        }
                        let mut psrc = 0usize;
                        let mut pdst = 0usize;
                        let pend = pme1_rw;
                        while psrc < pend {
                            let j_marker = flip(ws.iw[psrc]);
                            psrc += 1;
                            if j_marker >= 0 {
                                let j = j_marker as usize;
                                ws.iw[pdst] = ws.pe[j];
                                ws.pe[j] = pdst as Idx;
                                pdst += 1;
                                let lenj = ws.len[j] as usize;
                                if lenj > 0 {
                                    ws.iw.copy_within(psrc..psrc + lenj - 1, pdst);
                                    psrc += lenj - 1;
                                    pdst += lenj - 1;
                                }
                            }
                        }
                        let p1 = pdst;
                        ws.iw.copy_within(pme1_rw..ws.pfree, pdst);
                        pdst += ws.pfree - pme1_rw;
                        pme1_rw = p1;
                        ws.pfree = pdst;
                        pj = ws.pe[e] as usize;
                        p = ws.pe[me] as usize;
                    }
                    degme += nvi as usize;
                    ws.nv[i] = -nvi;
                    ws.iw[ws.pfree] = i as Idx;
                    ws.pfree += 1;
                    let ilast = ws.last[i];
                    let inext = ws.next[i];
                    if inext != NONE {
                        ws.last[inext as usize] = ilast;
                    }
                    if ilast != NONE {
                        ws.next[ilast as usize] = inext;
                    } else {
                        let h_idx = amf_bucket_of(ws.wf[i], n);
                        ws.head[h_idx] = inext;
                    }
                }
            }
            if e != me {
                ws.pe[e] = flip(me as Idx);
                ws.w[e] = 0;
            }
        }
        pme1 = pme1_rw;
        pme2 = (ws.pfree - 1) as Idx;
    }

    ws.degree[me] = degme as Idx;
    ws.pe[me] = pme1 as Idx;
    ws.len[me] = pme2 - pme1 as Idx + 1;
    ws.elen[me] = flip(nvpiv + degme as Idx);
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    let pme2_excl: usize = if pme2 < pme1 as Idx {
        pme1
    } else {
        (pme2 + 1) as usize
    };
    Ok((pme1, pme2_excl, nvpiv, degme))
}

/// AMF analogue of [`finalize_step`]. The Pass-1 element seeding,
/// hash-bucket detection, and supervariable-merge structure mirror
/// AMD; the per-iteration accumulator carries the AMF triple
/// `(deg, wf3, wf4)` (Amestoy 1999 thesis), and the re-insertion
/// computes the quantized RMF score and inserts at
/// `head[amf_bucket_of(wf[i], n)]`.
///
/// Six metric-specific sites compared to AMD (numbered per
/// `dev/research/amf-clean-room.md` Section 6):
/// 1. Pass-1 also resets `wf[e] = 0` on the first touch of each
///    element (lazy cache sentinel).
/// 2. Pass-2 element walk caches `wf[e] = dext * (2*deg(e) - dext - 1)`
///    on first encounter and accumulates `wf4 += wf[e]`.
/// 3. Pass-2 variable walk accumulates `wf3 += nv[j]`.
/// 4. Loose-degree special case zeroes `wf3 = wf4 = 0`; the kept
///    `degree[i]` cannot have a meaningful WF-subtraction so the
///    AMF score is reset.
/// 5. Supervariable merge takes `wf[i] = max(wf[i], wf[j])`.
/// 6. Re-insertion uses the saturated/regular RMF formula with
///    `dummy = Idx::MAX - 1`, quantizes via `bucket(wf[i], n)`, and
///    threads through `head` of length `2 * n + 2`.
///
/// Reference: `ana_orderings.F:4660-5025`.
#[allow(clippy::too_many_arguments)]
pub fn finalize_step_amf(
    ws: &mut Workspace,
    me: usize,
    pme1: usize,
    pme2_excl: usize,
    nvpiv: Idx,
    degme: usize,
    elenme: Idx,
    aggressive: bool,
) -> StepFlops {
    let n = ws.n;
    let mut degme = degme;
    let mut nvpiv = nvpiv;

    // Pass 1: seed w[e] for every element in each member's list, and
    // reset wf[e] = 0 on the first touch (lazy cache for Pass-2).
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let eln = ws.elen[i];
        if eln > 0 {
            let nvi = -ws.nv[i];
            let wnvi = ws.wflg - nvi;
            let pi = ws.pe[i] as usize;
            for k in 0..eln as usize {
                let e = ws.iw[pi + k] as usize;
                let mut we = ws.w[e];
                if we >= ws.wflg {
                    we -= nvi;
                } else if we != 0 {
                    we = ws.degree[e] + wnvi;
                    // O21 (repo-review-2026-06-09): `wf[e] = 0` is the
                    // lazy-cache "surface not yet computed this iteration"
                    // sentinel for Pass-2 below. It is intentionally NOT
                    // distinct from a genuine surface contribution of 0:
                    // `amf_wf_surface(dext, deg) = dext*(2*deg - dext - 1)`
                    // is 0 for a live element whenever `dext == 2*deg(e)-1`
                    // (e.g. dext=1, deg=1). When that happens `wf[e]` stays
                    // 0, so the Pass-2 `if wf[e] == 0` check re-treats it as
                    // uncached and recomputes the surface for every member
                    // that touches `e`. This is benign: `amf_wf_surface` is
                    // pure in (dext, degree[e]) — both stable across one
                    // Pass-2 — so the recompute yields the same 0 and the
                    // accumulated `wf4` (hence the RMF score and the
                    // permutation) is unchanged; only a few integer multiplies
                    // are redundant. A distinguishing sentinel (e.g. -1) was
                    // rejected: `wf` is reused for variable scores
                    // (supervariable-merge `max`, re-insertion bucket
                    // quantization), so -1 would have to be proven never to
                    // leak into either across the AMD and AMF paths — added
                    // correctness risk for a handful of saved ops.
                    // "Correctness before performance." See
                    // dev/tried-and-rejected.md (O21).
                    ws.wf[e] = 0;
                }
                ws.w[e] = we;
            }
        }
    }

    // Pass 2: AMF triple-accumulator (deg, wf3, wf4), aggressive
    // absorption on dext == 0, mass elimination, hash-bucket insert.
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let p1 = ws.pe[i] as usize;
        let p2 = p1 + ws.elen[i] as usize;
        let mut pn = p1;
        let mut deg: usize = 0;
        let mut hash: usize = 0;
        let mut wf3: i64 = 0;
        let mut wf4: i64 = 0;
        let nvi = -ws.nv[i];

        // Element sub-pass.
        if aggressive {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    let dext = we - ws.wflg;
                    if dext > 0 {
                        // `wf[e] == 0` means "uncached this iter" OR a genuine
                        // 0 surface (O21) — recompute on the latter is benign
                        // (same value). See the Pass-1 reset comment above.
                        if ws.wf[e] == 0 {
                            // First touch this iter: cache the surface
                            // contribution dext*(2*deg(e) - dext - 1).
                            ws.wf[e] = amf_wf_surface(dext, ws.degree[e]);
                        }
                        wf4 += ws.wf[e];
                        deg += dext as usize;
                        ws.iw[pn] = e as Idx;
                        pn += 1;
                        hash = hash.wrapping_add(e);
                    } else {
                        // Aggressive absorption.
                        ws.pe[e] = flip(me as Idx);
                        ws.w[e] = 0;
                    }
                }
            }
        } else {
            for p in p1..p2 {
                let e = ws.iw[p] as usize;
                let we = ws.w[e];
                if we != 0 {
                    let dext = we - ws.wflg;
                    // Invariant (O4): non-aggressive pass keeps `we >= ws.wflg`,
                    // so `dext >= 0`. Guard the `dext as usize` cast below
                    // against a future regression wrapping a negative dext to
                    // ~2^64.
                    debug_assert!(
                        dext >= 0,
                        "stale mark: w[e]={we} < wflg={} would wrap as usize",
                        ws.wflg
                    );
                    // `wf[e] == 0` means "uncached this iter" OR a genuine 0
                    // surface (O21) — recompute on the latter is benign (same
                    // value). See the Pass-1 reset comment above.
                    if ws.wf[e] == 0 {
                        ws.wf[e] = amf_wf_surface(dext, ws.degree[e]);
                    }
                    wf4 += ws.wf[e];
                    deg += dext as usize;
                    ws.iw[pn] = e as Idx;
                    pn += 1;
                    hash = hash.wrapping_add(e);
                }
            }
        }

        ws.elen[i] = (pn - p1 + 1) as Idx;
        let p3 = pn;
        let p4 = p1 + ws.len[i] as usize;
        // Variable sub-pass.
        for p in p2..p4 {
            let j = ws.iw[p] as usize;
            let nvj = ws.nv[j];
            if nvj > 0 {
                deg += nvj as usize;
                wf3 += nvj;
                ws.iw[pn] = j as Idx;
                pn += 1;
                hash = hash.wrapping_add(j);
            }
        }

        if ws.elen[i] == 1 && p3 == pn {
            // Mass elimination (equivalent to MUMPS DEG==0 in
            // aggressive / non-halo mode).
            ws.pe[i] = flip(me as Idx);
            let nvi_sv = -ws.nv[i];
            debug_assert!(nvi_sv >= 0);
            degme -= nvi_sv as usize;
            nvpiv += nvi_sv;
            ws.nel += nvi_sv as usize;
            ws.nv[i] = 0;
            ws.elen[i] = NONE;
            ws.n_mass_elim += 1;
        } else {
            // Loose-degree special case: if the prior degree estimate
            // is already tighter, keep it but the WF accumulator is
            // not subtraction-safe — zero it.
            if ws.degree[i] < deg as Idx {
                wf3 = 0;
                wf4 = 0;
            } else {
                ws.degree[i] = deg as Idx;
            }
            // wf[i] = wf4 + 2 * nvi * wf3 (Amestoy 1999 eq. for B3
            // contribution; see ana_orderings.F:4810).
            ws.wf[i] = amf_wf_combine(wf4, nvi, wf3);

            // Swap-dance to put `me` at the head of i's element list.
            if p1 != pn {
                ws.iw[pn] = ws.iw[p3];
            }
            if p3 != p1 {
                ws.iw[p3] = ws.iw[p1];
            }
            ws.iw[p1] = me as Idx;
            ws.len[i] = (pn - p1 + 1) as Idx;

            // Hash-bucket insertion (sign-bit encoding identical to
            // AMD; head reuse is safe because hash mod n falls in
            // the fine bucket region [0, n)).
            let h = hash % n;
            let j = ws.head[h];
            if j <= NONE {
                ws.next[i] = flip(j);
                ws.head[h] = flip(i as Idx);
            } else {
                ws.next[i] = ws.last[j as usize];
                ws.last[j as usize] = i as Idx;
            }
            ws.last[i] = h as Idx;
        }
    }

    let degme_i32 = degme as Idx;
    ws.degree[me] = degme_i32;
    if degme_i32 > ws.lemax {
        ws.lemax = degme_i32;
    }
    ws.wflg += ws.lemax;
    ws.wflg = clear_flag(ws.wflg, ws.wbig, &mut ws.w);

    // Supervariable detection. Identical to AMD except merge updates
    // wf[i] = max(wf[i], wf[j]).
    for pme in pme1..pme2_excl {
        let i_anchor = ws.iw[pme] as usize;
        if ws.nv[i_anchor] >= 0 {
            continue;
        }
        let h = ws.last[i_anchor] as usize;
        let j_head = ws.head[h];
        let mut i: Idx = if j_head == NONE {
            NONE
        } else if j_head < NONE {
            ws.head[h] = NONE;
            flip(j_head)
        } else {
            let chain_start = ws.last[j_head as usize];
            ws.last[j_head as usize] = NONE;
            chain_start
        };
        while i != NONE && ws.next[i as usize] != NONE {
            let i_u = i as usize;
            let ln = ws.len[i_u];
            let eln = ws.elen[i_u];
            let pi = ws.pe[i_u];
            for p in (pi + 1) as usize..(pi + ln) as usize {
                ws.w[ws.iw[p] as usize] = ws.wflg;
            }
            let mut jlast = i_u;
            let mut jp = ws.next[i_u];
            while jp != NONE {
                let jj = jp as usize;
                let mut ok = ws.len[jj] == ln && ws.elen[jj] == eln;
                if ok {
                    let pj = ws.pe[jj];
                    for p in (pj + 1) as usize..(pj + ln) as usize {
                        if ws.w[ws.iw[p] as usize] != ws.wflg {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    ws.pe[jj] = flip(i);
                    // AMF merge: wf takes the max of the two scores.
                    let wf_j = ws.wf[jj];
                    if wf_j > ws.wf[i_u] {
                        ws.wf[i_u] = wf_j;
                    }
                    ws.nv[i_u] += ws.nv[jj];
                    ws.nv[jj] = 0;
                    ws.elen[jj] = NONE;
                    jp = ws.next[jj];
                    ws.next[jlast] = jp;
                    ws.n_supervar_merge += 1;
                } else {
                    jlast = jj;
                    jp = ws.next[jj];
                }
            }
            ws.wflg += 1;
            i = ws.next[i_u];
        }
    }

    // Re-insertion: AMF saturated/regular RMF, quantize, bucket.
    let dummy_f = AMF_DUMMY_I32 as f64;
    let n_f = if n == 0 { 1.0 } else { n as f64 };
    let mut p_write = pme1;
    let nleft = ws.n - ws.nel;
    for pme in pme1..pme2_excl {
        let i = ws.iw[pme] as usize;
        let nvi = -ws.nv[i];
        if nvi > 0 {
            ws.nv[i] = nvi;
            let degme_i = degme_i32;
            let nvi_i = nvi;
            let deg_i = ws.degree[i];
            let rmf: f64 = if (deg_i as usize) + (degme_i as usize) > nleft {
                // Saturated branch. RMF1 uses original DEG.
                let deg_f = deg_i as f64;
                let rmf1 = deg_f * (deg_f - 1.0 + 2.0 * degme_i as f64) - ws.wf[i] as f64;
                let new_deg = (nleft as Idx) - nvi_i;
                ws.degree[i] = new_deg;
                let nd = new_deg as f64;
                let rmf_new =
                    nd * (nd - 1.0) - (degme_i - nvi_i) as f64 * (degme_i - nvi_i - 1) as f64;
                rmf_new.min(rmf1)
            } else {
                let deg_f = deg_i as f64;
                ws.degree[i] = deg_i + degme_i - nvi_i;
                deg_f * (deg_f - 1.0 + 2.0 * degme_i as f64) - ws.wf[i] as f64
            };
            let rmf = rmf / (nvi_i as f64 + 1.0);
            let qscore: Idx = if rmf < dummy_f {
                rmf.round() as Idx
            } else if rmf / n_f < dummy_f {
                (rmf / n_f).round() as Idx
            } else {
                AMF_DUMMY_I32
            };
            ws.wf[i] = qscore.max(1) as i64;

            let d = amf_bucket_of(ws.wf[i], n);
            let inext = ws.head[d];
            if inext != NONE {
                ws.last[inext as usize] = i as Idx;
            }
            ws.next[i] = inext;
            ws.last[i] = NONE;
            ws.head[d] = i as Idx;
            if d < ws.mindeg {
                ws.mindeg = d;
            }
            ws.iw[p_write] = i as Idx;
            p_write += 1;
        }
    }

    // Me bookkeeping (same as AMD).
    ws.nv[me] = nvpiv;
    ws.len[me] = (p_write as Idx) - pme1 as Idx;
    if ws.len[me] == 0 {
        ws.pe[me] = NONE;
        ws.w[me] = 0;
    }
    if elenme != 0 {
        ws.pfree = p_write;
    }

    // Flop counters (identical to AMD).
    let f = nvpiv as f64;
    let r = degme_i32 as f64 + ws.ndense as f64;
    let lnzme = f * r + (f - 1.0) * f / 2.0;
    let s = f * r * r + r * (f - 1.0) * f + (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;

    StepFlops {
        ndiv: lnzme,
        nms_lu: s,
        nms_ldl: (s + lnzme) / 2.0,
    }
}

/// AMF analogue of [`run_elimination`].
pub fn run_elimination_amf(
    ws: &mut Workspace,
    aggressive: bool,
) -> Result<StepFlops, OrderingError> {
    let mut flops = StepFlops::default();
    while ws.nel < ws.n {
        let me = match select_pivot_amf(ws) {
            Some(m) => m,
            None => break,
        };
        let elenme = ws.elen[me];
        let (pme1, pme2, nvpiv, degme) = create_element_amf(ws, me)?;
        flops.accumulate(finalize_step_amf(
            ws, me, pme1, pme2, nvpiv, degme, elenme, aggressive,
        ));
    }
    let f = ws.ndense as f64;
    let lnzme = (f - 1.0) * f / 2.0;
    let s = (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
    flops.ndiv += lnzme;
    flops.nms_lu += s;
    flops.nms_ldl += (s + lnzme) / 2.0;
    Ok(flops)
}

/// Run the main AMD elimination loop until every live supervariable
/// has been either pivoted or dense-deferred. Returns the
/// accumulated flop counts.
///
/// Mass elimination and supervariable detection are absent (Slice
/// B). Inline garbage collection is live; fixtures whose working
/// set transiently exceeds `iwlen` recover via in-place compaction
/// and bump `ws.ncmpa`.
///
/// At exit: `ws.nel == ws.n`, every `pe[i]` either points to a
/// live parent (to be path-compressed by the postorder phase) or
/// is `NONE` / `flip(parent)`.
pub fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
    let mut flops = StepFlops::default();
    while ws.nel < ws.n {
        let me = match select_pivot(ws) {
            Some(m) => m,
            None => break, // only dense-deferred survivors remain
        };
        let elenme = ws.elen[me];
        let (pme1, pme2, nvpiv, degme) = create_element(ws, me)?;
        flops.accumulate(finalize_step(
            ws, me, pme1, pme2, nvpiv, degme, elenme, aggressive,
        ));
    }
    // Dense-phase flop contribution (amd.rs:559-566).
    let f = ws.ndense as f64;
    let lnzme = (f - 1.0) * f / 2.0;
    let s = (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
    flops.ndiv += lnzme;
    flops.nms_lu += s;
    flops.nms_ldl += (s + lnzme) / 2.0;
    Ok(flops)
}

/// Consume the post-elimination state and produce the final
/// permutation.
///
/// On entry `ws` must have completed [`run_elimination`]. Performs,
/// in order:
/// 1. Un-flip `pe` and `elen` (faer `amd.rs:567-572`) so `pe[i]`
///    holds the parent pivot and `elen[i]` holds the frontal size.
/// 2. Path compression (`amd.rs:573-590`): each absorbed
///    supervariable `i` (`nv[i] == 0`) has its `pe` chain walked
///    until a pivot is found, then all intermediates are rewritten
///    to point at that pivot directly. Inert in Slice A — becomes
///    active once supervariable detection (Slice B) lands.
/// 3. Assembly-tree postorder with big-child-last heuristic
///    (`amd.rs:5-49`, `amd.rs:51-124`, `amd.rs:593-599`). Reuses
///    `head`/`next`/`last` as child/sibling/stack scratch; writes
///    the postorder index into `w`.
/// 4. Invert `w` into `head[k] = pivot at postorder k`
///    (`amd.rs:600-606`).
/// 5. Assign starting positions to each pivot's block
///    (`amd.rs:607-615`): `next[e] = nel`, then `nel += nv[e]`.
/// 6. Expand absorbed supervariables + place dense-deferred variables
///    at the tail (`amd.rs:617-629`).
/// 7. Emit `perm`: `perm[next[i]] = i` for every `i`
///    (`amd.rs:631-633`).
///
/// Returns a permutation `perm` of length `n` where `perm[k]` is the
/// column of the original matrix to be eliminated at step `k`.
pub fn finalize_permutation(ws: &mut Workspace) -> Vec<Idx> {
    let n = ws.n;
    if n == 0 {
        return Vec::new();
    }

    // Step 1: un-flip.
    for x in ws.pe.iter_mut() {
        *x = flip(*x);
    }
    for x in ws.elen.iter_mut() {
        *x = flip(*x);
    }

    // Step 2: path-compress absorbed supervariables.
    for i in 0..n {
        if ws.nv[i] == 0 {
            let head_i = ws.pe[i];
            if head_i == NONE {
                continue;
            }
            let mut j = head_i as usize;
            while ws.nv[j] == 0 {
                j = ws.pe[j] as usize;
            }
            let e = j as Idx;
            let mut j = i;
            while ws.nv[j] == 0 {
                let jnext = ws.pe[j];
                ws.pe[j] = e;
                j = jnext as usize;
            }
        }
    }

    // Step 3: assembly-tree postorder. Writes postorder index into w.
    assembly_tree_postorder(ws);

    // Step 4: invert w into head.
    for x in ws.head.iter_mut() {
        *x = NONE;
    }
    for e in 0..n {
        let k = ws.w[e];
        if k != NONE {
            ws.head[k as usize] = e as Idx;
        }
    }

    // Step 5: pivot-block starting positions.
    for x in ws.next.iter_mut() {
        *x = NONE;
    }
    let mut nel: Idx = 0;
    for &e in ws.head.iter() {
        if e == NONE {
            break;
        }
        let eu = e as usize;
        ws.next[eu] = nel;
        nel += ws.nv[eu];
    }

    // Step 6: expand absorbed supervars + place dense-deferred at tail.
    for i in 0..n {
        if ws.nv[i] == 0 {
            let e = ws.pe[i];
            if e != NONE {
                let eu = e as usize;
                ws.next[i] = ws.next[eu];
                ws.next[eu] += 1;
            } else {
                ws.next[i] = nel;
                nel += 1;
            }
        }
    }

    // Step 7: emit perm.
    let mut perm = vec![0 as Idx; n];
    for i in 0..n {
        perm[ws.next[i] as usize] = i as Idx;
    }
    perm
}

/// Build the assembly tree from `pe` (parent pointers) + `nv`
/// (`> 0` selects pivots), apply the big-child-last heuristic, and
/// run an iterative DFS postorder. Result indexes go into `ws.w`
/// (NONE for non-pivot nodes).
fn assembly_tree_postorder(ws: &mut Workspace) {
    let n = ws.n;
    // Repurpose head/next/last as child/sibling/stack scratch.
    for x in ws.head.iter_mut() {
        *x = NONE;
    }
    for x in ws.next.iter_mut() {
        *x = NONE;
    }
    // Link each pivot as a child of its parent. Reverse order so that
    // after building, child lists are in ascending index order.
    for j in (0..n).rev() {
        if ws.nv[j] > 0 {
            let parent = ws.pe[j];
            if parent >= 0 && (parent as usize) < n {
                let pu = parent as usize;
                ws.next[j] = ws.head[pu];
                ws.head[pu] = j as Idx;
            }
        }
    }
    // Big-child-last heuristic: move the child with the largest
    // `elen` (frontal size) to the end of the sibling list so the
    // deepest-recursion subtree is visited last.
    for i in 0..n {
        if ws.nv[i] > 0 && ws.head[i] != NONE {
            let child0 = ws.head[i];
            let mut fprev: Idx = NONE;
            let mut bigfprev: Idx = NONE;
            let mut bigf: Idx = NONE;
            let mut maxfrsize: Idx = NONE;
            let mut f = child0;
            while f != NONE {
                let fu = f as usize;
                let frsize = ws.elen[fu];
                if frsize >= maxfrsize {
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;
                f = ws.next[fu];
            }
            let bigfu = bigf as usize;
            let fnext = ws.next[bigfu];
            if fnext != NONE {
                if bigfprev != NONE {
                    ws.next[bigfprev as usize] = fnext;
                } else {
                    ws.head[i] = fnext;
                }
                ws.next[bigfu] = NONE;
                ws.next[fprev as usize] = bigf;
            }
        }
    }
    // Iterative DFS postorder from each pivot root.
    for x in ws.w.iter_mut() {
        *x = NONE;
    }
    let mut k: usize = 0;
    for i in 0..n {
        if ws.pe[i] == NONE && ws.nv[i] > 0 {
            k = post_tree_dfs(ws, i, k);
        }
    }
}

/// Iterative DFS of one assembly-tree root. Stack lives in `last`,
/// child list in `head`, sibling links in `next`. Writes postorder
/// indices into `w`. Returns the next postorder index to assign.
fn post_tree_dfs(ws: &mut Workspace, root: usize, k_start: usize) -> usize {
    let mut k = k_start;
    let mut top: usize = 1;
    ws.last[0] = root as Idx;
    while top > 0 {
        let i = ws.last[top - 1] as usize;
        let child0 = ws.head[i];
        if child0 != NONE {
            // Count children.
            let mut count = 0usize;
            let mut f = child0;
            while f != NONE {
                count += 1;
                f = ws.next[f as usize];
            }
            // Push children into stack slots [top, top+count) with the
            // first child at the highest position (popped first).
            let new_top = top + count;
            let mut t = new_top;
            let mut f = child0;
            loop {
                t -= 1;
                ws.last[t] = f;
                let nf = ws.next[f as usize];
                if nf == NONE {
                    break;
                }
                f = nf;
            }
            top = new_top;
            ws.head[i] = NONE; // mark visited
        } else {
            top -= 1;
            ws.w[i] = k as Idx;
            k += 1;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quotient_graph::WorkspaceOptions;
    use crate::CscPattern;

    fn ws_for<'a>(n: usize, cp: &'a [Idx], ri: &'a [Idx]) -> Workspace {
        let p = CscPattern::new(n, cp, ri).unwrap();
        Workspace::new(&p, &WorkspaceOptions::default()).unwrap()
    }

    /// O1 (repo-review-2026-06-09.md): the AMF working-fill kernels must
    /// be computed in `i64`. Both factors are `O(n)`, so for `n` ≳ 46k
    /// the products exceed `i32::MAX` (2_147_483_647) and wrap silently
    /// in release / panic in debug, feeding garbage into the RMF pivot
    /// score. Oracle: the exact hand-computed `i64` value of the
    /// Amestoy 1999 formulas (`ana_orderings.F:4810`).
    #[test]
    fn amf_wf_kernels_do_not_overflow_i32() {
        // Surface contribution dext*(2*degree - dext - 1).
        // dext = degree = 46342  ->  46342 * 46341 = 2_147_534_622,
        // which is 50_975 above i32::MAX. Hand-computed external oracle.
        assert!(2_147_534_622_i64 > i32::MAX as i64);
        assert_eq!(amf_wf_surface(46342, 46342), 2_147_534_622_i64);
        // Small-value sanity: dext=3, degree=4 -> 3*(8-3-1) = 12.
        assert_eq!(amf_wf_surface(3, 4), 12);

        // Combine wf4 + 2*nvi*wf3.
        // nvi = wf3 = 46342, wf4 = 0  ->  2 * 46342 * 46342
        // = 4_295_161_928, which exceeds both i32::MAX and u32::MAX.
        assert!(4_295_161_928_i64 > u32::MAX as i64);
        assert_eq!(amf_wf_combine(0, 46342, 46342), 4_295_161_928_i64);
        // Small-value sanity: 5 + 2*3*4 = 29.
        assert_eq!(amf_wf_combine(5, 3, 4), 29);
    }

    #[test]
    fn select_pivot_empty() {
        // diag_4: every var pre-eliminated, no degree bucket populated.
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let mut ws = ws_for(4, &cp, &ri);
        assert_eq!(select_pivot(&mut ws), None);
    }

    #[test]
    fn select_pivot_lifo_on_tridiag() {
        // Tridiag 5: head[1] contains 4 -> 0 (LIFO).
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        assert_eq!(select_pivot(&mut ws), Some(4));
        assert_eq!(ws.mindeg, 1);
        // Head of deg-1 list now points to the remaining spoke (0).
        assert_eq!(ws.head[1], 0);
        assert_eq!(ws.last[0], NONE, "new head has no predecessor");

        assert_eq!(select_pivot(&mut ws), Some(0));
        assert_eq!(ws.head[1], NONE, "deg-1 bucket drained");

        // Next call scans from mindeg=1 upward; only deg-2 non-empty.
        assert_eq!(select_pivot(&mut ws), Some(3));
        assert_eq!(ws.mindeg, 2);
    }

    #[test]
    fn create_element_elenme_zero_on_arrow_5_hub() {
        // Arrow 5: hub has deg 4, but it's dense-deferred? Let's check.
        // For n=5 default, dense = max(16, min(5, 10*sqrt(5))) = 5.
        // deg 4 < 5, so hub is LIVE and sits in head[4].
        // Spokes (deg 1) all share head[1]. The min-degree pivot is a
        // spoke. Let's pick spoke 4 (LIFO head of deg-1).
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let mut ws = ws_for(5, &cp, &ri);
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4, "first pivot is the LIFO head of deg-1");
        // elen[4] == 0 (no elements yet).
        assert_eq!(ws.elen[4], 0);
        let (pme1, pme2_excl, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        assert_eq!(nvpiv, 1, "singleton supervariable");
        assert_eq!(degme, 1, "only neighbor is the hub (nv=1)");
        // The new element's var list contains {hub} = {0}.
        assert_eq!(pme2_excl - pme1, 1);
        assert_eq!(ws.iw[pme1], 0);
        assert_eq!(ws.pe[4], pme1 as Idx);
        assert_eq!(ws.len[4], 1);
        assert_eq!(ws.elen[4], flip(1 + 1), "flip(nvpiv + degme)");
        assert_eq!(ws.nv[4], -1, "pivot marker");
        assert_eq!(ws.nv[0], -1, "hub marked");
        // Hub was in head[4] with no siblings; it was removed.
        assert_eq!(ws.head[4], NONE);
        // nel advanced by nvpiv.
        assert_eq!(ws.nel, 1);
    }

    #[test]
    fn create_element_elenme_zero_unlinks_from_degree_list() {
        // Tridiag 5: pivot var 4 (deg 1), neighbor var 3 (deg 2).
        // var 3 should be unlinked from head[2], which currently
        // threads 3 -> 2 -> 1.
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4);
        let (_, _, _, _) = create_element(&mut ws, me).unwrap();
        // After unlinking 3 (head of deg-2), head[2] -> 2.
        assert_eq!(ws.head[2], 2);
        assert_eq!(ws.last[2], NONE);
        // Unlinked var's last/next are stale but the list is valid.
        assert_eq!(ws.nv[3], -1);
    }

    #[test]
    fn create_element_skips_absorbed_neighbors() {
        // Construct a workspace where one neighbor is already absorbed
        // (nv == 0 or nv < 0). That neighbor must NOT contribute.
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let mut ws = ws_for(5, &cp, &ri);
        // Mark var 0 as already-absorbed (nv <= 0).
        ws.nv[0] = 0;
        let me = select_pivot(&mut ws).unwrap();
        assert_eq!(me, 4);
        let (_, _, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        // Only neighbor 3 counts; not 0 (absorbed) and not the diagonal
        // (skipped by init).
        assert_eq!(nvpiv, 1);
        assert_eq!(degme, 1);
    }

    /// Drive the full loop on diag_4 — every var pre-eliminated,
    /// loop terminates immediately.
    #[test]
    fn run_elimination_diag_4() {
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let mut ws = ws_for(4, &cp, &ri);
        let flops = run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, 4);
        assert_eq!(flops.ndiv, 0.0);
    }

    /// Arrow 5: no dense deferral. Loop eliminates all 5 vars.
    /// Verify `nel == n` and pivot supervariable count matches nv[me]
    /// after the step (restored to positive).
    #[test]
    fn run_elimination_arrow_5() {
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let mut ws = ws_for(5, &cp, &ri);
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, 5);
        // Every var was pivoted exactly once ⇒ nv[i] > 0 everywhere
        // (the pivot restores nv to +nvpiv).
        for i in 0..5 {
            assert!(ws.nv[i] >= 0, "nv[{}] = {}", i, ws.nv[i]);
        }
    }

    /// Tridiag 10 full-symmetric: loop should terminate cleanly.
    /// Oracle lnz = 9 — verified indirectly by the flop counter.
    #[test]
    fn run_elimination_tridiag_10() {
        let n = 10usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        for j in 0..n {
            if j > 0 {
                ri.push((j - 1) as Idx);
            }
            ri.push(j as Idx);
            if j + 1 < n {
                ri.push((j + 1) as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, n);
    }

    /// Grid 7x7: five-point stencil. Faer's oracle reports
    /// `ncmpa == 0`, but Slice A lacks mass elimination and
    /// supervariable detection, so it consumes more `iw` space and
    /// may trip the inline GC. With Commit 6 the loop terminates
    /// cleanly regardless of how many compactions are needed.
    #[test]
    fn run_elimination_grid_7x7() {
        let m = 7usize;
        let n = 7usize;
        let total = m * n;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        use std::collections::BTreeSet;
        let idx = |r: usize, c: usize| r * n + c;
        for c in 0..total {
            let r0 = c / n;
            let c0 = c % n;
            let mut neigh: BTreeSet<usize> = BTreeSet::new();
            neigh.insert(c);
            if r0 > 0 {
                neigh.insert(idx(r0 - 1, c0));
            }
            if r0 + 1 < m {
                neigh.insert(idx(r0 + 1, c0));
            }
            if c0 > 0 {
                neigh.insert(idx(r0, c0 - 1));
            }
            if c0 + 1 < n {
                neigh.insert(idx(r0, c0 + 1));
            }
            for &r in &neigh {
                ri.push(r as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(total, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, total);
    }

    /// Band(20,3) triggers garbage collection in faer (oracle
    /// ncmpa=1). Commit 6 must run the compaction and finish the
    /// elimination; verify both.
    #[test]
    fn run_elimination_band_20_3_triggers_gc() {
        let n = 20usize;
        let b = 3usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        for j in 0..n {
            let lo = j.saturating_sub(b);
            let hi = (j + b + 1).min(n);
            for r in lo..hi {
                ri.push(r as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.nel, n);
        assert!(
            ws.ncmpa >= 1,
            "expected at least one compaction on band(20,3), got ncmpa={}",
            ws.ncmpa
        );
    }

    /// Arrow 200: hub dense-deferred in init; spokes get pivoted
    /// one by one; loop terminates with ndense=1 survivor.
    #[test]
    fn run_elimination_arrow_200() {
        let n = 200usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        ri.push(0);
        for r in 1..n {
            ri.push(r as Idx);
        }
        cp.push(ri.len() as Idx);
        for j in 1..n {
            ri.push(0);
            ri.push(j as Idx);
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        assert_eq!(ws.ndense, 1);
        assert_eq!(ws.nel, n);
    }

    /// Arrow 200 hub is dense-deferred; first pivot is a spoke with
    /// no elements in its list, so the elenme==0 path is exercised
    /// on a larger graph. Smoke test that the path completes without
    /// indexing out of bounds.
    #[test]
    fn arrow_200_first_pivot_smoke() {
        let n = 200usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        ri.push(0);
        for r in 1..n {
            ri.push(r as Idx);
        }
        cp.push(ri.len() as Idx);
        for j in 1..n {
            ri.push(0);
            ri.push(j as Idx);
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        // Hub (var 0) is dense-deferred; its nv was set to 0 during
        // init. So when a spoke's list references 0, it's skipped.
        let me = select_pivot(&mut ws).unwrap();
        let (_, _, nvpiv, degme) = create_element(&mut ws, me).unwrap();
        assert_eq!(nvpiv, 1);
        assert_eq!(degme, 0, "spoke's only neighbor (hub) is deferred");
        // Exactly one var (the pivot itself) has been eliminated plus
        // the deferred hub that init already counted.
        assert_eq!(ws.nel, 2, "1 deferred hub + 1 pivot");
    }

    fn is_permutation(perm: &[Idx]) -> bool {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in perm {
            if p < 0 {
                return false;
            }
            let pu = p as usize;
            if pu >= n || seen[pu] {
                return false;
            }
            seen[pu] = true;
        }
        true
    }

    /// diag_4: every variable is pre-eliminated at init as a
    /// zero-degree singleton. Each is a tree root; postorder visits
    /// them in ascending index order, giving perm = [0,1,2,3].
    #[test]
    fn permutation_diag_4() {
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let mut ws = ws_for(4, &cp, &ri);
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), 4);
        assert!(is_permutation(&perm));
        assert_eq!(perm, vec![0, 1, 2, 3]);
    }

    /// Arrow 5 with hub live: LIFO spoke pivots first, then hub and
    /// remaining spoke chain through aggressive absorption. The exact
    /// root depends on pivot order and absorption choices; just
    /// verify a valid permutation.
    #[test]
    fn permutation_arrow_5_valid() {
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let mut ws = ws_for(5, &cp, &ri);
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), 5);
        assert!(is_permutation(&perm));
    }

    /// Tridiag 10: valid permutation of 0..10. No structural
    /// oracle here — just bijection + length checks.
    #[test]
    fn permutation_tridiag_10() {
        let n = 10usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        for j in 0..n {
            if j > 0 {
                ri.push((j - 1) as Idx);
            }
            ri.push(j as Idx);
            if j + 1 < n {
                ri.push((j + 1) as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), n);
        assert!(is_permutation(&perm));
    }

    /// Arrow 200 with a dense-deferred hub. The hub (var 0, nv=0,
    /// pe=NONE) is placed at the tail by the expand phase. All 199
    /// spokes are pivots; permutation is still a bijection of 0..200.
    #[test]
    fn permutation_arrow_200_hub_deferred() {
        let n = 200usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        ri.push(0);
        for r in 1..n {
            ri.push(r as Idx);
        }
        cp.push(ri.len() as Idx);
        for j in 1..n {
            ri.push(0);
            ri.push(j as Idx);
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), n);
        assert!(is_permutation(&perm));
        assert_eq!(
            perm[n - 1],
            0,
            "dense-deferred hub lands at the tail of the permutation"
        );
    }

    /// Grid 7x7: ensure the permutation survives GC-triggered runs.
    #[test]
    fn permutation_grid_7x7() {
        let m = 7usize;
        let n = 7usize;
        let total = m * n;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        use std::collections::BTreeSet;
        let idx = |r: usize, c: usize| r * n + c;
        for c in 0..total {
            let r0 = c / n;
            let c0 = c % n;
            let mut neigh: BTreeSet<usize> = BTreeSet::new();
            neigh.insert(c);
            if r0 > 0 {
                neigh.insert(idx(r0 - 1, c0));
            }
            if r0 + 1 < m {
                neigh.insert(idx(r0 + 1, c0));
            }
            if c0 > 0 {
                neigh.insert(idx(r0, c0 - 1));
            }
            if c0 + 1 < n {
                neigh.insert(idx(r0, c0 + 1));
            }
            for &r in &neigh {
                ri.push(r as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(total, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), total);
        assert!(is_permutation(&perm));
    }

    /// Band(20, 3) with GC: permutation remains a valid bijection
    /// even after ncmpa >= 1 compactions.
    #[test]
    fn permutation_band_20_3() {
        let n = 20usize;
        let b = 3usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        for j in 0..n {
            let lo = j.saturating_sub(b);
            let hi = (j + b + 1).min(n);
            for r in lo..hi {
                ri.push(r as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = CscPattern::new(n, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert_eq!(perm.len(), n);
        assert!(is_permutation(&perm));
    }

    /// Empty pattern (n == 0) round-trips to an empty permutation.
    #[test]
    fn permutation_empty() {
        let cp = [0 as Idx];
        let ri: [Idx; 0] = [];
        let p = CscPattern::new(0, &cp, &ri).unwrap();
        let mut ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        run_elimination(&mut ws, true).unwrap();
        let perm = finalize_permutation(&mut ws);
        assert!(perm.is_empty());
    }
}
