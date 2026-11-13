import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n)
    s = s.replace(old, new)

# named ops (JIT/PreJIT-able) under M1_NAMED_OPS
rep('''        ok!(GrB_IndexUnaryOp_new(&mut key, Some(key_op), t_u64(), t_i32(), t_u64()));
        ok!(GrB_IndexUnaryOp_new(&mut unkey, Some(unkey_op), t_u64(), t_u64(), t_u64()));
        ok!(GrB_IndexUnaryOp_new(&mut even, Some(even_op), t_bool(), t_i64(), t_i64()));
        ok!(GrB_IndexUnaryOp_new(&mut odd, Some(odd_op), t_bool(), t_i64(), t_i64()));''', '''        if env("M1_NAMED_OPS").is_some() {
            let kd = c"void m1_key (uint64_t *z, const int32_t *x, uint64_t i, uint64_t j, const uint64_t *y) { uint32_t h = ((uint32_t) j) ^ ((uint32_t) (*y)) ; h ^= h >> 16 ; h *= 0x85ebca6bu ; h ^= h >> 13 ; h *= 0xc2b2ae35u ; h ^= h >> 16 ; (*z) = (((uint64_t) (uint32_t) (*x)) << 32) | (uint64_t) h ; }";
            let ud = c"void m1_unkey (uint64_t *z, const uint64_t *x, uint64_t i, uint64_t j, const uint64_t *y) { uint32_t h = (uint32_t) (*x) ; h ^= h >> 16 ; h *= 0x7ed1b41du ; h ^= (h >> 13) ^ (h >> 26) ; h *= 0xa5cb9243u ; h ^= h >> 16 ; (*z) = (uint64_t) (h ^ ((uint32_t) (*y))) ; }";
            let ed = c"void m1_even (bool *z, const int64_t *x, uint64_t i, uint64_t j, const int64_t *y) { (*z) = ((j & 1) == 0) ; }";
            let od = c"void m1_odd (bool *z, const int64_t *x, uint64_t i, uint64_t j, const int64_t *y) { (*z) = ((j & 1) == 1) ; }";
            ok!(GxB_IndexUnaryOp_new(&mut key, Some(key_op), t_u64(), t_i32(), t_u64(), c"m1_key".as_ptr(), kd.as_ptr()));
            ok!(GxB_IndexUnaryOp_new(&mut unkey, Some(unkey_op), t_u64(), t_u64(), t_u64(), c"m1_unkey".as_ptr(), ud.as_ptr()));
            ok!(GxB_IndexUnaryOp_new(&mut even, Some(even_op), t_bool(), t_i64(), t_i64(), c"m1_even".as_ptr(), ed.as_ptr()));
            ok!(GxB_IndexUnaryOp_new(&mut odd, Some(odd_op), t_bool(), t_i64(), t_i64(), c"m1_odd".as_ptr(), od.as_ptr()));
            println!("OPS: named");
        } else {
            ok!(GrB_IndexUnaryOp_new(&mut key, Some(key_op), t_u64(), t_i32(), t_u64()));
            ok!(GrB_IndexUnaryOp_new(&mut unkey, Some(unkey_op), t_u64(), t_u64(), t_u64()));
            ok!(GrB_IndexUnaryOp_new(&mut even, Some(even_op), t_bool(), t_i64(), t_i64()));
            ok!(GrB_IndexUnaryOp_new(&mut odd, Some(odd_op), t_bool(), t_i64(), t_i64()));
        }''')

# JIT on with an isolated cache
rep('''fn verify() -> bool {
    env("M1_VERIFY").is_some()
}
''', '''fn verify() -> bool {
    env("M1_VERIFY").is_some()
}
/// M1_JIT=on: runtime JIT with the cache in M1_JIT_CACHE (never the shared ~/.SuiteSparse).
fn jit_setup() {
    match env("M1_JIT").as_deref() {
        Some("run") => {
            ok!(GrB_Global_set_INT32(GrB_GLOBAL, GxB_JIT_Control::GxB_JIT_RUN as i32, GxB_Option_Field::GxB_JIT_C_CONTROL as _));
            println!("JIT: RUN");
        }
        Some("on") => {
            let dir = env("M1_JIT_CACHE").expect("M1_JIT_CACHE");
            let c = std::ffi::CString::new(dir.clone()).unwrap();
            ok!(GrB_Global_set_String(GrB_GLOBAL, c.as_ptr() as *mut _, GxB_Option_Field::GxB_JIT_CACHE_PATH as _));
            ok!(GrB_Global_set_INT32(GrB_GLOBAL, GxB_JIT_Control::GxB_JIT_ON as i32, GxB_Option_Field::GxB_JIT_C_CONTROL as _));
            let mut ctl = 0i32;
            ok!(GrB_Global_get_INT32(GrB_GLOBAL, &mut ctl, GxB_Option_Field::GxB_JIT_C_CONTROL as _));
            println!("JIT: ON cache={dir} control={ctl}");
        }
        _ => {}
    }
}
''')
rep('''    if env("M1_JIT").as_deref() == Some("run") {
        ok!(GrB_Global_set_INT32(GrB_GLOBAL, GxB_JIT_Control::GxB_JIT_RUN as i32, GxB_Option_Field::GxB_JIT_C_CONTROL as _));
        println!("JIT: RUN");
    }
    let path = env("M1_GRAPH").expect("M1_GRAPH");
    let t = Instant::now();
    let s = load(&path);
    println!("graph {path}: n={} entries={} load={:.2}s", s.n, s.j.len(), t.elapsed().as_secs_f64());
    let a0 = to_grb(&s);
    let ops = Ops::new();
    let threads''', '''    jit_setup();
    let path = env("M1_GRAPH").expect("M1_GRAPH");
    let t = Instant::now();
    let s = load(&path);
    println!("graph {path}: n={} entries={} load={:.2}s", s.n, s.j.len(), t.elapsed().as_secs_f64());
    let a0 = to_grb(&s);
    let ops = Ops::new();
    let threads''')

# extra contraction variants
rep('''        aend();
        assert!(tup_i32(&ac2) == tup_i32(&ac), "AUDIT CHECK FAILED: contraction with C<!I> differs");
    }''', '''        aend();
        assert!(tup_i32(&ac2) == tup_i32(&ac), "AUDIT CHECK FAILED: contraction with C<!I> differs");
        drop(ac2);
        // (b) the official sequence without the MATERIALIZE wait on C*A
        abeg("contract_nowait");
        let tm3 = at.time("b_CA_nowait", || {
            let tm3 = M::new(t_i32(), nc, n);
            ok!(GrB_mxm(tm3.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, c.0, a.0, GrB_DESC_T0));
            tm3
        });
        let ac3 = at.time("b_CAC", || {
            let ac3 = M::new(t_i32(), nc, nc);
            ok!(GrB_mxm(ac3.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, tm3.0, c.0, null()));
            ac3
        });
        drop(tm3);
        at.time("b_offdiag", || {
            if !abrk("contract_b") {
                ok!(GrB_Matrix_select_INT64(ac3.0, null(), null(), GrB_OFFDIAG, ac3.0, 0, null()));
            }
            ac3.wait();
        });
        aend();
        assert!(tup_i32(&ac3) == tup_i32(&ac), "AUDIT CHECK FAILED: contraction without the C*A wait differs");
        drop(ac3);
        // (c) C (nc x n) packed from a host counting sort of the coarse ids: no transpose, Cw by dot
        let cah = ca.u64s().1; // the host already holds this map in a host numbering
        abeg("contract_hostC");
        let cm = at.time("c_hostC_pack", || {
            unsafe extern "C" {
                fn malloc(size: usize) -> *mut c_void;
            }
            let ncu = nc as usize;
            let nu = n as usize;
            let ap = unsafe { malloc((ncu + 1) * 8) as *mut u64 };
            let aj = unsafe { malloc(nu.max(1) * 8) as *mut u64 };
            let ax = unsafe { malloc(4) as *mut i32 };
            let (aps, ajs) = unsafe { (std::slice::from_raw_parts_mut(ap, ncu + 1), std::slice::from_raw_parts_mut(aj, nu.max(1))) };
            aps.iter_mut().for_each(|x| *x = 0);
            for &g in &cah {
                aps[g as usize + 1] += 1;
            }
            for k in 0..ncu {
                aps[k + 1] += aps[k];
            }
            let mut fill: Vec<u64> = aps[..ncu].to_vec();
            for (i, &g) in cah.iter().enumerate() {
                let f = &mut fill[g as usize];
                ajs[*f as usize] = i as u64;
                *f += 1;
            }
            if abrk("contract_c") {
                ajs.swap(0, 1);
            }
            unsafe { *ax = 1 };
            let cm = M::new(t_i32(), nc, n);
            let (mut app, mut ajp, mut axp) = (ap, aj, ax as *mut c_void);
            ok!(GxB_Matrix_pack_CSR(cm.0, &mut app, &mut ajp, &mut axp, ((ncu + 1) * 8) as u64, (nu.max(1) * 8) as u64, 4, true, false, null()));
            cm
        });
        let tm4 = at.time("c_hostC_CA", || {
            let tm4 = M::new(t_i32(), nc, n);
            ok!(GrB_mxm(tm4.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, cm.0, a.0, null()));
            tm4
        });
        let ac4 = at.time("c_hostC_CAC", || {
            let ac4 = M::new(t_i32(), nc, nc);
            ok!(GrB_mxm(ac4.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, tm4.0, c.0, null()));
            ac4
        });
        drop(tm4);
        at.time("c_hostC_offdiag", || {
            ok!(GrB_Matrix_select_INT64(ac4.0, null(), null(), GrB_OFFDIAG, ac4.0, 0, null()));
            ac4.wait();
        });
        let wc4 = at.time("c_hostC_Cw", || {
            let wc4 = V::new(t_i32(), nc);
            ok!(GrB_mxv(wc4.0, null(), null(), GrB_PLUS_TIMES_SEMIRING_INT32, cm.0, w.0, null()));
            wc4.wait();
            wc4
        });
        aend();
        assert!(tup_i32(&ac4) == tup_i32(&ac), "AUDIT CHECK FAILED: contraction with a packed C differs");
        assert!(wc4.i32s() == wc.i32s(), "AUDIT CHECK FAILED: Cw with a packed C differs");
    }''')
rep('''            "AUDITL n={n} nnz={} nc={nc} rounds={rounds} off_assign={:.6} off_unext={:.6} a_sel_true={:.6} a_assign_accum={:.6} a_emult_add={:.6} a_sel_false={:.6} off_CA={:.6} off_CAC={:.6} off_offdiag={:.6} a_CA_nowait={:.6} a_CAC_maskNotI={:.6}",''', '''            "AUDITL n={n} nnz={} nc={nc} rounds={rounds} off_assign={:.6} off_unext={:.6} a_sel_true={:.6} a_assign_accum={:.6} a_emult_add={:.6} a_sel_false={:.6} off_CA={:.6} off_CAC={:.6} off_offdiag={:.6} a_CA_nowait={:.6} a_CAC_maskNotI={:.6} off_cbuild={:.6} off_Cw={:.6} b_CA_nowait={:.6} b_CAC={:.6} b_offdiag={:.6} c_hostC_pack={:.6} c_hostC_CA={:.6} c_hostC_CAC={:.6} c_hostC_offdiag={:.6} c_hostC_Cw={:.6}",''')
rep('''            h("a_CA_nowait"),
            h("a_CAC_maskNotI")
        );''', '''            h("a_CA_nowait"),
            h("a_CAC_maskNotI"),
            g("c_build"),
            g("Cw"),
            h("b_CA_nowait"),
            h("b_CAC"),
            h("b_offdiag"),
            h("c_hostC_pack"),
            h("c_hostC_CA"),
            h("c_hostC_CAC"),
            h("c_hostC_offdiag"),
            h("c_hostC_Cw")
        );''')
open(p, 'w').write(s)
print('ok')
