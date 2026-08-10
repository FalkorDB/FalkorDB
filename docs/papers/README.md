# Papers

Long-form design write-ups of FalkorDB internals, in arXiv preprint form.

| File | Subject |
| --- | --- |
| `tensor.tex` | `Tensor`, the per-relationship-type edge store (`graph/src/graph/graphblas/tensor.rs`) — inline edge identifiers with sentinel promotion, the MVCC delta algebra for value-carrying layers, the square-root fold policy and its measured constants, the measured
multiplicity sweep against the C container-per-cell implementation, both engines
measured again at the data-structure boundary, and what those two grains refute, the two hazard classes that come from building on a non-blocking GraphBLAS runtime, and the Lean 4 mechanisation of the invariant set. |

## Building

Each `.tex` is self-contained: the bibliography is inline (`thebibliography`),
so no `.bib` file or `bibtex` run is needed. Two passes resolve the
cross-references.

```sh
pdflatex -interaction=nonstopmode tensor.tex
pdflatex -interaction=nonstopmode tensor.tex
```

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

- Cite the artifact precisely: file path, branch, and commit. These documents
  describe code at a point in time and go stale silently otherwise. When a
  revision moves the citation, say what the old one pointed at — a reader
  holding the earlier PDF needs to know the branch was renamed, not guess.
- Keep claims falsifiable. Where a document reports costs that were derived
  rather than measured, it says so. Where it reports a machine-checked result,
  it also says what the mechanisation does *not* cover, since the gap between a
  model and the code it models is the part no proof can close.
- Build products (`.pdf`, `.aux`, `.log`, `.out`) are not committed.
- A revision is not done until `pdflatex` runs twice with no undefined
  references and no LaTeX warnings. If no TeX is installed locally,
  `docker run --rm -v "$PWD:/w" -w /w texlive/texlive:latest pdflatex
  -interaction=nonstopmode tensor.tex` needs no host toolchain.

## Open work

[`OPEN_WORK.md`](OPEN_WORK.md) expands the paper's Future work section into
actionable form: for each open item, the design, how to know it worked, and what
could go wrong. It also records what the evaluation deliberately does not settle,
so that scope does not have to be re-derived from the paper's prose.
