# Known gaps at packaging

1. The supplied manuscript references `figure_3_uncertainty_transport.pdf`.
   That exact asset is absent from the research tree. The retained older
   `figure_3_upstream_downstream_transfer.*` uses a different presentation;
   it has not been substituted or relabeled. Building `manuscript/main.tex`
   needs the correct Figure 3. Other retained figures also require a final
   visual comparison with the supplied manuscript before submission.
2. The supplied manuscript discusses supplementary Section S7 and HiVT
   tables, while the retained `SUPPLEMENTARY_APPENDIX.md` predates the
   HiVT extension. The extension's report, protocol, data contract, CSV tables,
   and summary JSON are included; an updated submission supplement was not
   supplied or reconstructed.
3. Original upstream probability-head files and their authority exports
   had been lost before the HiVT extension. Its `recovery_v1` contains
   independently restored heads, not byte-identical copies of the original
   CV/MLP study's heads. Do not interchange these versions in reported results.
4. Large derived data files and the pretrained detector are external inputs.
   `EXCLUDED_ARTIFACTS.csv` groups excluded files by source directory,
   subdirectory, extension, count, and byte size. No data download or training
   was performed while creating this repository.
5. Historical freeze manifests retain their original source hashes and
   machine paths. They describe the experiments as executed, not the new
   package location. They must not be edited to make a new run appear to be
   an original frozen run. The HiVT startup preflight expects an upstream Git
   checkout; the bundled source snapshot has its commit recorded in
   `THIRD_PARTY_NOTICES.md` instead of retaining nested `.git` internals.

Only the graphics search path differs between `manuscript/main.tex` and the
unmodified attachment. Scientific claims, tables, references, authors, and
acknowledgments have not been edited for packaging.
