# d5-judge: the recount behind the D5 (b) verdict

**Backs:** `../../d5-b-verdict.md`, the source it labels **J**.

**Came from:** the scratch folder `d5judge`.

## Files

| File | What it is |
|---|---|
| `pair.py LOG…` | Per graph and variant (`mm-reuse`, `mm-reuse-capstop`, `mm-reuse-ge`), counts the seeds whose arcs, triangles and height are at or below SHEM's and R's (`mm`). It also prints the medians. |
| `rall3.py LOG…` | For `mm`, `mm-reuse`, `mm-reuse-capstop` and `mm-reuse-ge`: the seeds, out of 5, where all three metrics are at or below SHEM. |
| `diag_broken.diff` | The deliberate break: one TX capstop height (seed 1) raised from 664 to 999 in a copy of stage 2's `diag.txt`. |

## Run

The inputs are the `RESULT` lines of the matching harness's stage-2 logs, `out/main.txt` and `out/diag.txt`. They are in `../m1-matching/stage2-results/raw-logs.tar.xz`.

```bash
tar -xJf ../m1-matching/stage2-results/raw-logs.tar.xz        # gives out/main.txt, out/diag.txt, …
python3 rall3.py out/main.txt out/diag.txt
python3 pair.py out/main.txt out/diag.txt

# the break: TX capstop must drop from 5/5 to 4/5
cp out/diag.txt diag_broken.txt && patch diag_broken.txt diag_broken.diff
python3 rall3.py out/main.txt diag_broken.txt
```

## Expected output

The verdict's "Seeds, all 3 ≤ feral" column:
- R 12/15;
- plain reuse (b) 1/15;
- capstop 13/15.

With `diag_broken.txt`, TX capstop goes from 5/5 to 4/5.

**Checked on 2026-11-13** from this copy:
- `rall3.py` printed `mm` 4/5, 4/5, 4/5; `mm-reuse` 0/5, 1/5, 0/5; and `mm-reuse-capstop` 4/5, 4/5, 5/5 (CA, PA, TX).
- The patched file was byte-identical to the original `diag_broken.txt`, and gave TX 4/5.
