import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n)
    s = s.replace(old, new)

# cap probe: rowscale then colscale with PLUS_PLUS, no eWiseMult
rep('''    // (1b) keys by a user positional op.''', '''    let mut at = Timer::default();
    if let Some(s_off) = mcap.as_ref().filter(|_| audit("cap")) {
        abeg("cap");
        let s2 = at.time("q_cap_rowcol", || {
            let mut d = null();
            ok!(GrB_Matrix_diag(&mut d, w.0, 0));
            let d = M(d);
            let s2 = M::new(t_i32(), n, n);
            ok!(GrB_mxm(s2.0, null(), null(), GxB_PLUS_FIRST_INT32, d.0, a.0, null()));
            let sr = if abrk("cap") { GxB_PLUS_FIRST_INT32 } else { GxB_PLUS_PLUS_INT32 };
            ok!(GrB_mxm(s2.0, null(), null(), sr, s2.0, d.0, null()));
            ok!(GrB_Matrix_select_INT32(s2.0, null(), null(), GrB_VALUELE_INT32, s2.0, cap, null()));
            s2.wait();
            s2
        });
        aend();
        assert!(tup_i32(&s2) == tup_i32(s_off), "AUDIT CHECK FAILED: cap mask by row/col scale differs");
    }
    // (1b) keys by a user positional op.''')
rep('''    let mut rounds = 0;
    let mut matched = 0u64;
    let mut at = Timer::default();
    loop {''', '''    let mut rounds = 0;
    let mut matched = 0u64;
    // zmatch probe: Z = mate overlaid with this round's proposals; the mutual test reads Z(Z)
    let mut zv = if audit("zmatch") && !brk("nomutual") { Some(idx.dup()) } else { None };
    loop {''')
rep('''        let mate_before = if audit("match") && !brk("nomutual") { Some((mate.dup(), mate.dup())) } else { None };''', '''        if let Some(z) = zv.as_mut() {
            abeg("zmatch");
            let z2 = at.time("z_add", || {
                let z2 = V::new(t_u64(), n);
                ok!(GrB_Vector_eWiseAdd_BinaryOp(z2.0, null(), null(), GrB_SECOND_UINT64, z.0, y.0, null()));
                z2.wait();
                z2
            });
            *z = z2;
            let qz = at.time("z_gather", || {
                let qz = V::new(t_u64(), n);
                ok!(GxB_Vector_extract_Vector(qz.0, null(), null(), z.0, z.0, null()));
                qz.wait();
                qz
            });
            let mz = at.time("z_eq", || {
                let mz = V::new(t_bool(), n);
                ok!(GrB_Vector_eWiseMult_BinaryOp(mz.0, y.0, null(), GrB_EQ_UINT64, qz.0, idx.0, GrB_DESC_RS));
                mz.wait();
                mz
            });
            aend();
            let tb = |v: &V| {
                let mut k = v.nvals();
                let mut i = vec![0u64; k as usize];
                let mut x = vec![false; k as usize];
                ok!(GrB_Vector_extractTuples_BOOL(i.as_mut_ptr(), x.as_mut_ptr(), &mut k, v.0));
                (i, x)
            };
            assert!(tb(&mz) == tb(&m), "AUDIT CHECK FAILED: zmatch mutual test differs (round {rounds})");
        }
        let mate_before = if audit("match") && !brk("nomutual") { Some((mate.dup(), mate.dup())) } else { None };''')
rep('''    let round_mate = if verify() { Some(mate.u64s().1) } else { None };''', '''    if let Some(z) = zv.take() {
        abeg("zfinal");
        let m2 = at.time("z_final", || {
            // stale proposals (proposed, never matched) go back to self: mate = Z, except where Z(Z) != idx
            let qz = V::new(t_u64(), n);
            ok!(GxB_Vector_extract_Vector(qz.0, null(), null(), z.0, z.0, null()));
            let eq = V::new(t_bool(), n);
            ok!(GrB_Vector_eWiseMult_BinaryOp(eq.0, null(), null(), GrB_EQ_UINT64, qz.0, idx.0, null()));
            let st = V::new(t_bool(), n);
            ok!(GrB_Vector_select_BOOL(st.0, null(), null(), GrB_VALUEEQ_BOOL, eq.0, false, null()));
            let m2 = V::new(t_u64(), n);
            if abrk("zmatch") {
                ok!(GrB_Vector_apply(m2.0, null(), null(), GrB_IDENTITY_UINT64, z.0, null()));
            } else {
                let si = V::new(t_u64(), n);
                ok!(GrB_Vector_apply_IndexOp_UINT64(si.0, null(), null(), GrB_ROWINDEX_INT64, st.0, 0, null()));
                ok!(GrB_Vector_eWiseAdd_BinaryOp(m2.0, null(), null(), GrB_SECOND_UINT64, z.0, si.0, null()));
            }
            m2.wait();
            m2
        });
        aend();
        assert!(m2.u64s() == mate.u64s(), "AUDIT CHECK FAILED: zmatch final mate differs");
    }
    let round_mate = if verify() { Some(mate.u64s().1) } else { None };''')
# print the new probe totals in AUDITL
rep('''c_hostC_offdiag={:.6} c_hostC_Cw={:.6}",''', '''c_hostC_offdiag={:.6} c_hostC_Cw={:.6} off_cap={:.6} q_cap_rowcol={:.6} off_full={:.6} off_extract={:.6} off_eq={:.6} z_add={:.6} z_gather={:.6} z_eq={:.6} z_final={:.6}",''')
rep('''            h("c_hostC_offdiag"),
            h("c_hostC_Cw")
        );''', '''            h("c_hostC_offdiag"),
            h("c_hostC_Cw"),
            g("cap"),
            h("q_cap_rowcol"),
            g("m_full"),
            g("m_extract"),
            g("m_eq"),
            h("z_add"),
            h("z_gather"),
            h("z_eq"),
            h("z_final")
        );''')
# the old `let mut at` inside the rounds section must go (now declared earlier)
open(p, 'w').write(s)
print('ok')
