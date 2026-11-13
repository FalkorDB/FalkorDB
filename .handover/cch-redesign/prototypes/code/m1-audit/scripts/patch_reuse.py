import sys
p = sys.argv[1]
s = open(p).read()
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n)
    s = s.replace(old, new)

# burble per depth
rep('''fn mat_tuples(m: &M)''', '''/// Library burble only at the depths listed in M1_BURBLE_DEPTHS (e.g. "0,1,13").
fn set_burble(d: u32) {
    if let Some(s) = env("M1_BURBLE_DEPTHS") {
        let on = s.split(',').any(|x| x.parse::<u32>().ok() == Some(d));
        ok!(GxB_Global_Option_set_INT32(GxB_Option_Field::GxB_BURBLE as _, on as i32));
        println!("@@DEPTH {d} burble={}", on as u8);
        flush();
    }
}

/// AUDIT: delete the rows and columns of `drop` from a dup of `src` in place (zombies), insert
/// `ins` as pending tuples, one wait. Returns (matrix, deleted entries, seconds by phase).
fn inplace_delete_insert(
    src: &M,
    drop_rows: &[u64],
    new_dim: u64,
    ins: Option<&(Vec<u64>, Vec<u64>, Vec<i32>)>,
) -> (M, u64, [f64; 4]) {
    let p = {
        let mut d = null();
        ok!(GrB_Matrix_dup(&mut d, src.0));
        M(d)
    };
    p.wait();
    let dim = p.nrows();
    let t0 = Instant::now();
    let mut ndel = 0u64;
    if !drop_rows.is_empty() {
        let rows = M::new(t_i32(), drop_rows.len() as u64, dim);
        ok!(GrB_Matrix_extract(rows.0, null(), null(), p.0, drop_rows.as_ptr(), drop_rows.len() as u64, GrB_ALL, dim, null()));
        let mut k = rows.nvals();
        let mut ri = vec![0u64; k as usize];
        let mut rj = vec![0u64; k as usize];
        ok!(GrB_Matrix_extractTuples_INT32(ri.as_mut_ptr(), rj.as_mut_ptr(), null(), &mut k, rows.0));
        for q in 0..k as usize {
            let (i, j) = (drop_rows[ri[q] as usize], rj[q]);
            ok!(GrB_Matrix_removeElement(p.0, i, j));
            ok!(GrB_Matrix_removeElement(p.0, j, i));
        }
        ndel = k;
    }
    let t_del = t0.elapsed().as_secs_f64();
    if new_dim > dim {
        ok!(GrB_Matrix_resize(p.0, new_dim, new_dim));
    }
    if let Some((ii, jj, xx)) = ins {
        for q in 0..ii.len() {
            ok!(GrB_Matrix_setElement_INT32(p.0, xx[q], ii[q], jj[q]));
        }
    }
    let t_ins = t0.elapsed().as_secs_f64();
    p.wait();
    let t_all = t0.elapsed().as_secs_f64();
    (p, ndel, [t_del, t_ins - t_del, t_all - t_ins, t_all])
}

/// AUDIT probe for B2: the same child level from the parent's matrix by in-place deletion.
fn audit_b2_inplace(
    par: &HLevel,
    bk: &Book,
    rs: Option<&M>,
    official: &M,
    t: &Timer,
) {
    if !audit("inplace") {
        return;
    }
    abeg("b2inplace");
    let ins = rs.map(mat_tuples); // a host merge would hold these already (PLUS-reduced)
    let th = Instant::now();
    let mut keep = vec![false; par.dim];
    for &q in &bk.clean {
        keep[q as usize] = true;
    }
    let mut dl: Vec<u64> = par.live.iter().filter(|&&q| !keep[q as usize]).map(|&q| q as u64).collect();
    if abrk("inplace") && !dl.is_empty() {
        dl.remove(0);
    }
    let host = th.elapsed().as_secs_f64();
    let (p, ndel, ph) = inplace_delete_insert(&par.a, &dl, bk.dim as u64, ins.as_ref());
    aend();
    let a = mat_tuples(&p);
    let b = mat_tuples(official);
    assert!(a == b, "AUDIT CHECK FAILED: in-place B2 level differs (nnz {} vs {})", a.0.len(), b.0.len());
    let off: f64 = ["m_build", "mask_mxm", "mask_vals", "resize", "s_merge"].iter().map(|k| t.0.get(k).copied().unwrap_or(0.0)).sum();
    println!(
        "AUDITB2 pd={} dim={} nnzP={} nnzQ={} drop_rows={} del={ndel} ins={} official={off:.6} probe={:.6} host={host:.6} del_t={:.6} ins_t={:.6} wait_t={:.6}",
        par.dim,
        bk.dim,
        par.a.nvals(),
        official.nvals(),
        dl.len(),
        ins.as_ref().map_or(0, |x| x.0.len()),
        host + ph[3],
        ph[0],
        ph[1],
        ph[2]
    );
}

fn mat_tuples(m: &M)''')

# B2: probe at both return points
rep('''    let xs: Vec<u64> = bk.strad_x.iter().filter(|&&x| !(masked && (bk.up[x as usize] as u64) < pd)).map(|&x| x as u64).collect();
    if xs.is_empty() {
        return (sm, 0);
    }''', '''    let xs: Vec<u64> = bk.strad_x.iter().filter(|&&x| !(masked && (bk.up[x as usize] as u64) < pd)).map(|&x| x as u64).collect();
    if xs.is_empty() {
        audit_b2_inplace(par, bk, None, &sm, t);
        return (sm, 0);
    }''')
rep('''    drop(tr);
    let q = t.time("s_merge", || {
        let q = M::new(t_i32(), bk.dim as u64, bk.dim as u64);
        ok!(GrB_Matrix_eWiseAdd_BinaryOp(q.0, null(), null(), GrB_PLUS_INT32, sm.0, rs.0, null()));
        q.wait();
        q
    });
    (q, rin)''', '''    drop(tr);
    let q = t.time("s_merge", || {
        let q = M::new(t_i32(), bk.dim as u64, bk.dim as u64);
        ok!(GrB_Matrix_eWiseAdd_BinaryOp(q.0, null(), null(), GrB_PLUS_INT32, sm.0, rs.0, null()));
        q.wait();
        q
    });
    audit_b2_inplace(par, bk, Some(&rs), &q, t);
    (q, rin)''')

# burble switch at depth 0 and every depth; fine-split in-place probe
rep('''            println!("LOAD threads={th} run={run} load1={:.2}", load1());
            // depth 0''', '''            println!("LOAD threads={th} run={run} load1={:.2}", load1());
            set_burble(0);
            let mut prev_live: Vec<u32> = (0..s.n as u32).collect();
            // depth 0''')
rep('''                let cells_act = lab.active.iter().filter(|&&a| a).count() as u64;''', '''                let cells_act = lab.active.iter().filter(|&&a| a).count() as u64;
                set_burble(d);''')
rep('''                let split_t: f64 = st.0.values().sum();
                let q0 = Rc::new(q0);''', '''                let split_t: f64 = st.0.values().sum();
                if audit("inplace") {
                    abeg("splitinplace");
                    let th = Instant::now();
                    let mut keep = vec![false; s.n];
                    for &v in &live {
                        keep[v as usize] = true;
                    }
                    let mut dl: Vec<u64> = prev_live.iter().filter(|&&v| !keep[v as usize]).map(|&v| v as u64).collect();
                    if abrk("splitinplace") && !dl.is_empty() {
                        dl.remove(0);
                    }
                    let host = th.elapsed().as_secs_f64();
                    let (p, ndel, ph) = inplace_delete_insert(&prev_q0, &dl, n, None);
                    aend();
                    let a = mat_tuples(&p);
                    let b = mat_tuples(&q0);
                    assert!(a == b, "AUDIT CHECK FAILED: in-place fine split differs (depth {d}, nnz {} vs {})", a.0.len(), b.0.len());
                    println!(
                        "AUDITSPLIT threads={th} run={run} depth={d} nnz_prev={} nnz_new={} drop_rows={} del={ndel} official={split_t:.6} probe={:.6} host={host:.6} del_t={:.6} wait_t={:.6}",
                        prev_q0.nvals(),
                        q0.nvals(),
                        dl.len(),
                        host + ph[3],
                        ph[0],
                        ph[2]
                    );
                }
                prev_live = live.clone();
                let q0 = Rc::new(q0);''')
open(p, 'w').write(s)
print("ok")
