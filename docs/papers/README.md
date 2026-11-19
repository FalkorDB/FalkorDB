# Papers

Long-form design write-ups of FalkorDB internals, in arXiv preprint form.

| File | Subject |
| --- | --- |
| `tensor.tex` | `Tensor`, the per-relationship-type edge store (`graph/src/graph/graphblas/tensor.rs`): inline edge identifiers with sentinel promotion, the MVCC delta algebra for value-carrying layers, the square-root fold policy and its measured constants, a multiplicity sweep against the C container-per-cell implementation, both structures measured again at the data-structure boundary, the two hazard classes that come from building on a non-blocking GraphBLAS runtime, and the Lean 4 mechanisation in `proofs/versioned_matrix`. |
| `pattern.tex` | The same idea with the edges taken out: *inline-first with sentinel promotion* as a reusable pattern for any sparse store whose cells usually hold one value. The two preconditions, the six obligations an adopter inherits (one of which should be **declined** rather than discharged), the ten-state cost of putting values in cells, a second instantiation sketched and then argued against, and the measurement traps, generalised. Reads standalone; `tensor.tex` is its worked instantiation. |

## Building

Each `.tex` is self-contained: the bibliography is inline (`thebibliography`),
so no `.bib` file or `bibtex` run is needed. Two passes resolve the
cross-references.

```sh
pdflatex -interaction=nonstopmode tensor.tex
pdflatex -interaction=nonstopmode tensor.tex
```

`pattern.tex` builds the same way and needs a smaller package set — no TikZ,
`algorithm` or `listings`.

Required TeX Live packages beyond `texlive-latex-base`:
`texlive-latex-recommended`, `texlive-latex-extra`, `texlive-pictures`
(TikZ), `texlive-fonts-recommended`, and `texlive-science` (`algorithm`,
`algpseudocode`). On Debian/Ubuntu:

```sh
apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-pictures texlive-fonts-recommended texlive-science
```

## Conventions

- Cite the artifact precisely: file path and commit. These documents describe
  code at a point in time and go stale silently otherwise. Measurements say
  which build they were taken on; proofs are cited by theorem name in
  `proofs/`, so a reader can check them with `#print axioms`.
- Keep claims falsifiable. Where a document reports costs that were derived
  rather than measured, it says so. Where it reports a machine-checked result,
  it also says what the mechanisation does *not* cover, since the gap between a
  model and the code it models is the part no proof can close.
- Build products (`.pdf`, `.aux`, `.log`, `.out`) are not committed.
- A change is not done until `pdflatex` runs twice with no undefined
  references, no LaTeX warnings and no overfull boxes. If no TeX is installed locally,
  `docker run --rm -v "$PWD:/w" -w /w texlive/texlive:latest pdflatex
  -interaction=nonstopmode tensor.tex` needs no host toolchain.
