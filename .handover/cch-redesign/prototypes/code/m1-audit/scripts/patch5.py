import sys
p, q = sys.argv[1], sys.argv[2]
s = open(p).read()
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n)
    s = s.replace(old, new)
rep('''/// Seconds per named call, summed within a level.
#[derive(Default, Clone)]
struct Timer(BTreeMap<&'static str, f64>);''', '''unsafe extern "C" {
    fn proc_pid_rusage(
        pid: i32,
        flavor: i32,
        buffer: *mut c_void,
    ) -> i32;
    fn getpid() -> i32;
}
/// (instructions, cycles) of this process so far: RUSAGE_INFO_V4 ri_instructions / ri_cycles.
fn instr() -> (u64, u64) {
    let mut b = [0u64; 40];
    let rc = unsafe { proc_pid_rusage(getpid(), 4, b.as_mut_ptr() as *mut c_void) };
    assert_eq!(rc, 0, "proc_pid_rusage");
    (b[31], b[32])
}

/// Seconds per named call, summed within a level; .1 holds (instructions, cycles).
#[derive(Default, Clone)]
struct Timer(BTreeMap<&'static str, f64>, BTreeMap<&'static str, (u64, u64)>);''')
rep('''        let t = Instant::now();
        let r = f();
        let el = t.elapsed().as_secs_f64();
        *self.0.entry(name).or_default() += el;''', '''        let i0 = instr();
        let t = Instant::now();
        let r = f();
        let el = t.elapsed().as_secs_f64();
        let i1 = instr();
        *self.0.entry(name).or_default() += el;
        let e = self.1.entry(name).or_default();
        e.0 += i1.0 - i0.0;
        e.1 += i1.1 - i0.1;''')
# AUDITL: instructions of the key official/probe groups
rep('''    if !at.0.is_empty() {
        let g = |k: &str| t.0.get(k).copied().unwrap_or(0.0);''', '''    if !at.0.is_empty() {
        let gi = |k: &str| t.1.get(k).map_or(0, |x| x.0);
        let hi = |k: &str| at.1.get(k).map_or(0, |x| x.0);
        println!(
            "AUDITI n={n} unext={} sel_false={} assign_full_extract_eq={} z={} CA_CAC_offdiag={} b={} cbuild_Cw={} hostC={} cap={} q_cap={}",
            gi("m_unext"),
            hi("a_sel_false"),
            gi("m_assign") + gi("m_full") + gi("m_extract") + gi("m_eq"),
            hi("z_add") + hi("z_gather") + hi("z_eq") + hi("z_final"),
            gi("CA") + gi("CAC'") + gi("offdiag"),
            hi("b_CA_nowait") + hi("b_CAC") + hi("b_offdiag"),
            gi("c_build") + gi("Cw"),
            hi("c_hostC_pack") + hi("c_hostC_CA") + hi("c_hostC_CAC") + hi("c_hostC_offdiag") + hi("c_hostC_Cw"),
            gi("cap"),
            hi("q_cap_rowcol")
        );
        let g = |k: &str| t.0.get(k).copied().unwrap_or(0.0);''')
open(p, 'w').write(s)

s = open(q).read()
def rep2(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n)
    s = s.replace(old, new)
rep2('''    let host = th.elapsed().as_secs_f64();
    let (p, ndel, ph) = inplace_delete_insert(&par.a, &dl, bk.dim as u64, ins.as_ref());''', '''    let host = th.elapsed().as_secs_f64();
    let i0 = instr();
    let (p, ndel, ph) = inplace_delete_insert(&par.a, &dl, bk.dim as u64, ins.as_ref());
    let i1 = instr();''')
rep2('''    let off: f64 = ["m_build", "mask_mxm", "mask_vals", "resize", "s_merge"].iter().map(|k| t.0.get(k).copied().unwrap_or(0.0)).sum();''', '''    let off: f64 = ["m_build", "mask_mxm", "mask_vals", "resize", "s_merge"].iter().map(|k| t.0.get(k).copied().unwrap_or(0.0)).sum();
    let offi: u64 = ["m_build", "mask_mxm", "mask_vals", "resize", "s_merge"].iter().map(|k| t.1.get(k).map_or(0, |x| x.0)).sum();
    // the probe's dup (untimed) sits inside i0..i1: subtract a dup's instructions measured alone
    let dupi = {
        let a0 = instr();
        let mut d = null();
        ok!(GrB_Matrix_dup(&mut d, par.a.0));
        let d = M(d);
        d.wait();
        let a1 = instr();
        drop(d);
        a1.0 - a0.0
    };
    println!("AUDITI2 kind=b2 official_instr={offi} probe_instr={}", (i1.0 - i0.0).saturating_sub(dupi));''')
rep2('''                    let (p, ndel, ph) = inplace_delete_insert(&prev_q0, &dl, n, None);''', '''                    let i0 = instr();
                    let (p, ndel, ph) = inplace_delete_insert(&prev_q0, &dl, n, None);
                    let i1 = instr();
                    let dupi = {
                        let a0 = instr();
                        let mut dd = null();
                        ok!(GrB_Matrix_dup(&mut dd, prev_q0.0));
                        let dd = M(dd);
                        dd.wait();
                        let a1 = instr();
                        drop(dd);
                        a1.0 - a0.0
                    };
                    let offi: u64 = st.1.values().map(|x| x.0).sum();
                    println!("AUDITI2 kind=split depth={d} official_instr={offi} probe_instr={}", (i1.0 - i0.0).saturating_sub(dupi));''')
open(q, 'w').write(s)
print('ok')
