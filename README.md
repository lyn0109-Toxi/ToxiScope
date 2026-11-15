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

## NTR evidence workspace

Select **Toxicity Evidence Review** in the sidebar. Upload permitted documents,
classify their source, and review proposed assertions before evaluation. Structural
screens are separate context, not observed toxicity facts. Processing occurs on
the app server, with no external AI calls; this is not browser-local processing.
Exports include document text, so treat them as confidential. Session state is
not a persistent project store.

This is a fixed-rule research prototype focused on early hepatotoxicity. Other
endpoints are unevaluable (R0). R0–R5 are uncalibrated evidence-role labels, not
safety predictions or regulatory acceptance. Name lookup is local by default;
PubChem/NCI requests require query-specific opt-in.
