# Papers

Long-form design write-ups of FalkorDB internals, in arXiv preprint form.

| File | Subject |
| --- | --- |
| `tensor.tex` | `Tensor`, the per-relationship-type edge store (`graph/src/graph/graphblas/tensor.rs`, `main-rs`) — inline edge identifiers with sentinel promotion, the MVCC delta algebra for value-carrying layers, and the two hazard classes that come from building on a non-blocking GraphBLAS runtime. |

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
  describe code at a point in time and go stale silently otherwise.
- Keep claims falsifiable. Where a document reports costs that were derived
  rather than measured, it says so.
- Build products (`.pdf`, `.aux`, `.log`, `.out`) are not committed.
