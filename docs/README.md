# Documentation

| Document | Read it for |
|---|---|
| [showcase.md](showcase.md) | the three demonstrations, each with its scope and links into the evidence |
| [g17-technical-reference.pdf](g17-technical-reference.pdf) | how the machine, its instruction set, the compiler and both execution paths work; every statement cites the evidence record (LaTeX source in [tex/](tex/), `make -C docs/tex` rebuilds it) |
| [g17-tensorops-machine-model.md](g17-tensorops-machine-model.md) | the evidence record: every measurement in numbered sections ("MM 25.211"), corrections and failed controls kept in place |
| [g17-tensorops-accelerator-recon.md](g17-tensorops-accelerator-recon.md) | the earlier tensor-unit investigation the evidence record cites by section number ("section 132") |
| [findings.md](findings.md) | episodes where a measurement overturned a claim |
| [execution-validation.md](execution-validation.md) | the rules for dispatching programs this project authored, and why they exist |
| [figures/](figures/) | the README's figures, drawn from the receipts by `tools/g17showcasefigs.py` |

The evidence record and the recon document are the research record as written. They cite files, tools and notes
that this curated release does not carry (the release keeps the receipts behind its public claims; see
[RELEASE.md](../RELEASE.md)); every such link now leads to an entry in [OMITTED.md](OMITTED.md) that names the target
and why it was left out. They use the project's earlier name, triad, in places.
