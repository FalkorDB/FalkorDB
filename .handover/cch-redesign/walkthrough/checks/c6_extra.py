"""Independent check 6: the v15 commit's work units; F4's front and the arcs it reaches;
B_gil headroom. Reuses c4's functions by import of c4 (my own check), not walkthrough.py."""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c4", os.path.join(HERE, "c4_metric_queries.py"))
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        c4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c4)
    except SystemExit:
        pass
ho, hn = c4.from_json("14"), c4.from_json("15")
ro = {x: r for x, r in hn.maps["rank_of"]}
fr = [tuple(sorted((ro[12], ro[13])))]
M, pops, ch, work = c4.scoped(hn, c4.W(15), ho.metric, fr)
print("v15 SET 12->13: pops", pops, "changed", ch, "work", work, "equals published:", all(M[i] == hn.metric[i] for i in hn.arcs.values()))
h12, h13 = c4.from_json("12"), c4.from_json("13")
r7 = {x: r for x, r in h13.maps["rank_of"]}[7]
for (lo, hi), i in sorted(h13.arcs.items()):
    if r7 in (lo, hi):
        print("  F4 arc", (lo, hi), "id", i, "v12", h12.metric[i][:2], "v13", h13.metric[i][:2])
