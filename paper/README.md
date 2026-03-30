# Paper Source

This directory contains the live ACL-style manuscript source for *When Accuracy Hides Path Dependence: Criterion Fidelity in LLM Judges*.

## Contents

- `main.tex` - manuscript entrypoint
- `figures/` - figure PDFs used directly by the paper
- `acl.sty`, `acl_natbib.bst` - bundled style dependencies
- `latexmkrc` - local build configuration
- `build/` - generated LaTeX output directory

## Build

Preferred command from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File paper/build_paper.ps1
```

Or, from this directory:

```powershell
.\build_paper.ps1
```

The bundled `latexmkrc` writes auxiliary files and the compiled PDF to `build/`. The build script also refreshes `When_Accuracy_Hides_Path_Dependence_Criterion_Fidelity_in_LLM_Judges_Ali_Uyar.pdf` so the reader-facing copy stays current. Generated output in `build/` is not tracked in the public repository.
