# ToxiScope AI

Rule-based structural alert screening and regulatory evidence review with
impurity evidence, degradation profiling, and draft narrative support.

The SAR screen matches a limited set of SMARTS rules. It does not implement a
trained statistical model, calibrated mutagenicity probabilities, or a second
independent QSAR method. No alert is an unknown finding within this limited
screen, not proof of safety, negative mutagenicity, or ICH M7 Class 5. A shared
drug-substance alert does not establish control or Class 4. The app leaves the
ICH M7 class unassigned for qualified review. Local historical dossier assertions
are retained as unverified source records, not independently confirmed evidence.
Workflow checks assess software output completeness and traceability, not
scientific validation or regulatory acceptance.

## Streamlit Cloud Deployment

Use these settings when creating the app on Streamlit Community Cloud:

- Repository: `lyn0109-Toxi/ToxiScope`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python dependencies: `requirements.txt`

The cloud entry point imports `toxiscope_app.py`, so the local app and deployed
app use the same ToxiScope runtime.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Software regression checks

```bash
pip install -r requirements-dev.txt
bash scripts/validate.sh
```

Set `PYTHON` to a virtual environment's Python executable when needed. The suite
includes headless Streamlit interaction tests for SAR, BE, and NTR. These are
software regression checks, not scientific validation or regulatory acceptance.

## Core Modules

- `core/regulatory.py`: compatibility API for app and tests
- `core/qsar.py`: limited SMARTS structural screening; legacy statistical APIs remain for compatibility without model predictions
- `core/evidence.py`: evidence objects and source traceability
- `core/compendial.py`: USP/EP/DMF-style known impurity context
- `core/degradation.py`: predicted degradation product assessment
- `core/harness.py`: software workflow checks and worker-report style manifest
- `core/reporting.py`: draft narrative generation with screening limitations
- `core/bioequivalence.py`: descriptive dissolution f2 comparison and parametric simulation; clinical BE and regulatory acceptance are not established

## Dissolution profile comparison

The BE workspace provides a descriptive f2 calculation and an assumption-based
parametric simulation from supplied dissolution summary statistics. A high f2,
low variation, or simulation percentile does not establish clinical
bioequivalence, biowaiver eligibility, or regulatory acceptance. FDA/ICH links
are background references; they do not validate this implementation or any
product-specific use. The strategy dashboard keeps the result under review.

Inputs must include the required time, reference/test mean percentage, SD, and
sample-size columns with finite numeric values. Times must be unique and
strictly positive; their ordering is normalized without removing points. Mean
percentages must lie between 0 and 100, SDs must be nonnegative, and sample sizes
must be integers of at least two, constant within each arm. Reference and test
arms may have different sample sizes. Missing or invalid values are rejected
rather than imputed or silently removed. Values above 100, including analytical
recoveries, are outside this model's supported input range and are not clamped.
The calculation requires at least three time points. Sample sizes below 12
produce an applicability warning. These software checks do not establish that
a study design meets product-specific regulatory requirements.

The simulation draws each sample mean from a normal distribution with the
entered mean and standard error SD/sqrt(n). It does not resample observed units
and is not an empirical bootstrap. Reference/test arms and sampling times are
assumed independent. Longitudinal within-unit correlations cannot be recovered
from summary data, and simulated values are not clipped to 0–100. Percentiles
describe this assumed simulation distribution; they are not calibrated
confidence intervals, BE probabilities, or probabilities of regulatory success.

Results use schema `dissolution-f2-simulation.v2` and retain assumptions,
limitations, the fixed seed, backend, and software versions for review. A
repeated calculation with the same inputs and backend is reproducible within
the recorded software environment. Python and R use different random streams,
so the same seed does not make their draws identical. Selecting R requires a
successful R execution; an R failure is reported without silently switching to
Python. Use the result's `to_dict()` method for export; it excludes legacy
CI/bootstrap/probability/FDA attribute aliases. Clinical BE remains unestablished
in exported results.

The CV display uses a labeled legacy heuristic of 20% at the first retained
point and 10% at later points. It is a variability flag, not a universal
FDA/EMA acceptance policy. Review the applicable guidance and product-specific
requirements separately. Background sources include the
[FDA dissolution guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/dissolution-testing-immediate-release-solid-oral-dosage-forms)
and the [EMA investigation of bioequivalence guideline, revision 1](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-bioequivalence-rev1_en.pdf).

SD inputs are sample standard deviations in percentage points; they are not
standard errors or CV values. The summary-normal simulation does not implement
the individual-observation/Expected-f2 bootstrap discussed in the
[EMA comparative-dissolution Q&A](https://www.ema.europa.eu/en/scientific-guidelines/clinical-pharmacology-pharmacokinetics-questions-answers)
(sections 3.11 and 3.13 on the reviewed page). No EMA compliance claim is made.

## NTR evidence workspace

Select **Toxicity Evidence Review** in the sidebar. Upload permitted documents,
classify their source, and review the proposed assertions. The project review
report lists possible disagreements, differences in study scope, source issues,
and gaps in source-linked intake coverage for expert review. It does not decide
which result is correct, establish toxicity, or validate a method.

Assertions retain their original extracted value, proposed or revised value,
review status and note, document ID and hash, location, and source excerpt.
Evidence kind separates observations, predictions, and context. Comparison
scope records the subject, endpoint, system, exposure, timepoint, and study ID;
missing scope remains uncertain, and differences in explicit scope are not
confirmed contradictions. Proposed and rejected assertions remain visible in
the review record. Only reviewed assertions linked to classified customer
evidence count as reviewed source coverage. External references and unclassified
documents do not become reviewed customer evidence through approval alone.

Predictions and structural screens remain separate from observed NAM results.
Manual and default assessment inputs are unverified and do not establish
source-linked coverage. Missing evidence means unknown or unreviewed evidence;
it does not mean a negative result or absence of toxicity. No endpoint-specific
scientific adjudication has been validated.

Processing occurs on the app server, with no external AI calls; this is not
browser-local processing. Exports include source text and review metadata, so
treat them as confidential. Session state is not a persistent project store.

Downloads include the review report even when optional legacy evaluation is
blocked. Different project/compound workspaces remain separate within the session;
unmatched legacy-session evidence is preserved separately until its project
binding can be reviewed. Pairwise comparison is bounded and explicitly reports
when further claims require manual inspection. Enabled-stream intake prompts are
internal review aids, not a validated or mandatory regulatory checklist.

This is a fixed-rule research prototype focused on early hepatotoxicity. Other
endpoints are unevaluable (R0). R0–R5 are uncalibrated evidence-role labels, not
safety predictions or regulatory acceptance. Name lookup is local by default;
PubChem/NCI requests require query-specific opt-in.
