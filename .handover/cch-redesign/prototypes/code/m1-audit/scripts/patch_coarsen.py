import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:60], n)
    s = s.replace(old, new)

# helpers
rep('''fn verify() -> bool {
    env("M1_VERIFY").is_some()
}
''', '''fn verify() -> bool {
    env("M1_VERIFY").is_some()
}
/// AUDIT probes (untimed by the official timers): M1_AUDIT=match,contract,inplace
fn audit(k: &str) -> bool {
    env("M1_AUDIT").is_some_and(|s| s.split(',').any(|x| x == k))
}
fn abrk(k: &str) -> bool {
    env("M1_AUDIT_BREAK").as_deref() == Some(k)
}
fn tup_i32(m: &M) -> (Vec<u64>, Vec<u64>, Vec<i32>) {
    let mut n = m.nvals();
    let mut i = vec![0u64; n as usize];
    let mut j = vec![0u64; n as usize];
    let mut x = vec![0i32; n as usize];
    ok!(GrB_Matrix_extractTuples_INT32(i.as_mut_ptr(), j.as_mut_ptr(), x.as_mut_ptr(), &mut n, m.0));
    let mut p: Vec<usize> = (0..i.len()).collect();
    p.sort_unstable_by_key(|&k| (i[k], j[k]));
    (p.iter().map(|&k| i[k]).collect(), p.iter().map(|&k| j[k]).collect(), p.iter().map(|&k| x[k]).collect())
}
fn vidx(v: &V) -> Vec<u64> {
    let mut n = v.nvals();
    let mut i = vec![0u64; n as usize];
    ok!(GrB_Vector_extractTuples_BOOL(i.as_mut_ptr(), null(), &mut n, v.0));
    i
}
fn abeg(k: &str) {
    if burble() {
        println!("@@AUDIT begin {k}");
        flush();
    }
}
fn aend() {
    if burble() {
        println!("@@AUDIT end");
        flush();
    }
}
''')

# per-level audit timer + matching probe
rep('''    let mut rounds = 0;
    let mut matched = 0u64;
    loop {''', '''    let mut rounds = 0;
    let mut matched = 0u64;
    let mut at = Timer::default();
    loop {''')
rep('''        let np = y.nvals();
        if np == 0 {
            break;
        }''', '''        let np = y.nvals();
        if burble() {
            println!("@@NP round={rounds} n={n} proposers={np}");
        }
        if np == 0 {
            break;
        }''')
rep('''        t.time("m_assign", || {
            ok!(GrB_Vector_assign(mate.0, m.0, null(), y.0, GrB_ALL, n, null()));
            mate.wait();
        });''', '''        let mate_before = if audit("match") && !brk("nomutual") { Some((mate.dup(), mate.dup())) } else { None };
        t.time("m_assign", || {
            ok!(GrB_Vector_assign(mate.0, m.0, null(), y.0, GrB_ALL, n, null()));
            mate.wait();
        });''')
rep('''        let newly = np - u2.nvals();
        u = u2;''', '''        if let Some((mb, mb2)) = mate_before {
            abeg("match");
            let want = !abrk("match");
            let ms = at.time("a_sel_true", || {
                let ms = V::new(t_bool(), n);
                ok!(GrB_Vector_select_BOOL(ms.0, null(), null(), GrB_VALUEEQ_BOOL, m.0, want, null()));
                ms.wait();
                ms
            });
            at.time("a_assign_accum", || {
                ok!(GrB_Vector_assign(mb.0, ms.0, GrB_SECOND_UINT64, y.0, GrB_ALL, n, GrB_DESC_S));
                mb.wait();
            });
            at.time("a_emult_add", || {
                let ym = V::new(t_u64(), n);
                ok!(GrB_Vector_eWiseMult_BinaryOp(ym.0, null(), null(), GrB_FIRST_UINT64, y.0, ms.0, null()));
                ok!(GrB_Vector_eWiseAdd_BinaryOp(mb2.0, null(), null(), GrB_SECOND_UINT64, mb2.0, ym.0, null()));
                mb2.wait();
            });
            let u3 = at.time("a_sel_false", || {
                let u3 = V::new(t_bool(), n);
                ok!(GrB_Vector_select_BOOL(u3.0, null(), null(), GrB_VALUEEQ_BOOL, m.0, false, null()));
                u3.wait();
                u3
            });
            aend();
            let off = mate.u64s();
            assert!(mb.u64s() == off, "AUDIT CHECK FAILED: match (assign with accum) differs from mate");
            assert!(mb2.u64s() == off, "AUDIT CHECK FAILED: match (emult+add) differs from mate");
            assert!(vidx(&u3) == u2.u64s().0, "AUDIT CHECK FAILED: match unmatched pattern differs");
        }
        let newly = np - u2.nvals();
        u = u2;''')

# contraction probe, after offdiag/Cw, before the checks
rep('''    // checks (cheap, always on; untimed)
    let diag = M::new(t_i32(), nc, nc);''', '''    if audit("contract") && ct() {
        abeg("contract");
        let tm2 = at.time("a_CA_nowait", || {
            let tm2 = M::new(t_i32(), nc, n);
            ok!(GrB_mxm(tm2.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, c.0, a.0, GrB_DESC_T0));
            tm2
        });
        let ac2 = at.time("a_CAC_maskNotI", || {
            let iv = V::new(t_bool(), nc);
            ok!(GrB_Vector_assign_BOOL(iv.0, null(), null(), true, GrB_ALL, nc, null()));
            let mut dg = null();
            ok!(GrB_Matrix_diag(&mut dg, iv.0, 0));
            let dg = M(dg);
            let ac2 = M::new(t_i32(), nc, nc);
            let (mk, ds) = if abrk("contract") { (null(), null()) } else { (dg.0, unsafe { GrB_DESC_SC }) };
            ok!(GrB_mxm(ac2.0, mk, null(), GrB_PLUS_TIMES_SEMIRING_INT32, tm2.0, c.0, ds));
            ac2.wait();
            ac2
        });
        aend();
        assert!(tup_i32(&ac2) == tup_i32(&ac), "AUDIT CHECK FAILED: contraction with C<!I> differs");
    }
    if !at.0.is_empty() {
        let g = |k: &str| t.0.get(k).copied().unwrap_or(0.0);
        let h = |k: &str| at.0.get(k).copied().unwrap_or(0.0);
        println!(
            "AUDITL n={n} nnz={} nc={nc} rounds={rounds} off_assign={:.6} off_unext={:.6} a_sel_true={:.6} a_assign_accum={:.6} a_emult_add={:.6} a_sel_false={:.6} off_CA={:.6} off_CAC={:.6} off_offdiag={:.6} a_CA_nowait={:.6} a_CAC_maskNotI={:.6}",
            a.nvals(),
            g("m_assign"),
            g("m_unext"),
            h("a_sel_true"),
            h("a_assign_accum"),
            h("a_emult_add"),
            h("a_sel_false"),
            g("CA"),
            g("CAC'"),
            g("offdiag"),
            h("a_CA_nowait"),
            h("a_CAC_maskNotI")
        );
    }
    // checks (cheap, always on; untimed)
    let diag = M::new(t_i32(), nc, nc);''')
open(p, 'w').write(s)
print("ok")
