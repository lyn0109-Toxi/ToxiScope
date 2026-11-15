# NTR core origin
The initial nora/ modules were copied from the owner's repository:
https://github.com/lyn0109-Toxi/ToxiGuard-NORA
Commit: 6d5e75476188cb99b9884483502599d605a4ed09
Integration date: 2026-11-14.

Included: models, engine, ai_credibility, evidence, assertions, evidence_review,
branding and package version constants. No license expansion or third-party
data redistribution permission is implied. The source repository supplied no
LICENSE file; this owner-directed integration into the public ToxiScope repository does not
grant third-party licensing or data redistribution rights. No Platform repository
license is transferred by this integration.

Integration code: ntr_bridge.py and ntr_panel.py. The original engine rules were
not changed by this integration.

Local evidence-review changes extend nora/evidence.py, nora/assertions.py, and
nora/evidence_review.py with assertion kind and comparison scope, retained
extraction provenance, and a project-level review report. The report flags
possible disagreements, scope differences, source issues, and intake coverage
gaps for expert review. It preserves document identity/hash, location, excerpt,
original and proposed values, and reviewer actions. These checks assess source
linkage and review state; they do not independently verify the source's
scientific claims or adjudicate toxicity.

Observation, prediction, and context assertions remain distinct. Manual/default
inputs and external or unclassified documents do not establish reviewed
customer-evidence coverage. Missing evidence remains unknown. The review report
does not provide endpoint validation, calibrated toxicity predictions,
regulatory acceptance, or authorization to reduce animal testing.
