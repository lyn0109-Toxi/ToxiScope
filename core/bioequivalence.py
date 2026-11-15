import csv
import math
import hashlib
import json
import platform
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


DEFAULT_DISSOLUTION_PROFILE = pd.DataFrame(
    [
        {"Time (min)": 5, "Reference Mean (%)": 32.0, "Reference SD": 4.5, "Reference n": 12, "Test Mean (%)": 30.0, "Test SD": 5.0, "Test n": 12},
        {"Time (min)": 10, "Reference Mean (%)": 51.0, "Reference SD": 5.2, "Reference n": 12, "Test Mean (%)": 49.0, "Test SD": 5.6, "Test n": 12},
        {"Time (min)": 15, "Reference Mean (%)": 68.0, "Reference SD": 6.0, "Reference n": 12, "Test Mean (%)": 66.0, "Test SD": 6.4, "Test n": 12},
        {"Time (min)": 30, "Reference Mean (%)": 86.0, "Reference SD": 5.5, "Reference n": 12, "Test Mean (%)": 84.0, "Test SD": 6.0, "Test n": 12},
        {"Time (min)": 45, "Reference Mean (%)": 93.0, "Reference SD": 4.0, "Reference n": 12, "Test Mean (%)": 91.0, "Test SD": 4.7, "Test n": 12},
    ]
)
FDA_BE_GUIDANCE_SOURCES = [
    {
        "Guidance": "M13A Bioequivalence for Immediate-Release Solid Oral Dosage Forms",
        "Status": "Final",
        "Date": "October 2024",
        "Use in platform": "Background on clinical PK bioequivalence; this dissolution calculator does not implement or validate an M13A BE study.",
        "URL": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m13a-bioequivalence-immediate-release-solid-oral-dosage-forms",
    },
    {
        "Guidance": "Dissolution Testing of Immediate Release Solid Oral Dosage Forms",
        "Status": "Final",
        "Date": "August 1997",
        "Use in platform": "Background on dissolution-profile comparison; product-specific applicability and any biowaiver are not established by this calculator.",
        "URL": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/dissolution-testing-immediate-release-solid-oral-dosage-forms",
    },
    {
        "Guidance": "FDA Dissolution Methods Database",
        "Status": "FDA database",
        "Date": "Quarterly update",
        "Use in platform": "Product-specific method search when no USP dissolution method is available.",
        "URL": "https://www.fda.gov/drugs/drug-approvals-and-databases/dissolution-methods-database",
    },
    {
        "Guidance": "Dissolution Testing and Acceptance Criteria for IR Drug Products Containing High Solubility Drug Substances",
        "Status": "Final",
        "Date": "August 2018",
        "Use in platform": "High-solubility IR product dissolution acceptance strategy.",
        "URL": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/dissolution-testing-and-acceptance-criteria-immediate-release-solid-oral-dosage-form-drug-products",
    },
]


RESULT_SCHEMA = "dissolution-f2-simulation.v2"
_UNSET = object()
REQUIRED_COLUMNS = ["Time (min)", "Reference Mean (%)", "Reference SD", "Reference n", "Test Mean (%)", "Test SD", "Test n"]
VARIABILITY_POLICY = "Legacy EMA Appendix I screening: first nonzero sampled point CV <=20%, later points <=10%; not a universal current regulatory eligibility rule."
SIMULATION_ASSUMPTIONS = [
    "SD means the sample standard deviation of unit-level percent dissolved in percentage points, not a standard error or CV. n counts dissolution units, not clinical subjects.",
    "Independent normal sample means at each supplied time and arm: N(entered mean, entered sample SD squared / entered n).",
    "Entered means, sample SDs and n are treated as fixed plug-in parameters. Their own estimation uncertainty is not modeled.",
    "No within-unit longitudinal covariance or cross-arm dependence is available from summary input; independence is an assumption, not observed study evidence.",
    "Normal draws are unbounded: values below 0% or above 100% and nonmonotonic simulated profiles are possible. Draws are not clipped, truncated or silently discarded.",
    "The ordinary f2 formula is applied to every simulated pair of mean profiles. Linear/type-7 empirical percentiles summarize this simulation only.",
]
LIMITATIONS = [
    "Descriptive dissolution comparison and optional parametric simulation only; no clinical bioequivalence or regulatory acceptance is established.",
    "Summary-data normal simulation is not unit-level empirical bootstrap, an Expected-f2 bootstrap, a calibrated confidence interval or a probability of clinical bioequivalence.",
    "This simulation does not implement the EMA comparative-dissolution bootstrap procedure or establish its applicability; prespecified protocol and individual-unit observations are not supplied or verified.",
    "The simulated fraction above 50 depends on the assumed distribution, entered SD/n, seed, iteration count and numerical backend. It is not a posterior probability or an acceptance test.",
    "Protocol, unit-level profiles, medium, apparatus, agitation, batches, analytical validation, sampling completeness, product-specific requirements and biowaiver conditions are not verified by this calculator.",
    "No points are removed automatically. Time-zero exclusion, sampling plan and any >85% tail-point selection must be justified prospectively by qualified review, not selected to improve f2.",
    "The named CV policy is a conservative legacy screen; current guidance and product-specific requirements can use different early-time definitions. Passing it is not regulatory qualification.",
    "Python and R use different random-number streams. A shared seed supports repeatability within the recorded backend/version, not identical simulated values across backends or external validation.",
]


class ProfileValidationError(ValueError):
    """An input is outside this calculator's documented numeric model."""


class BackendUnavailableError(RuntimeError):
    """The explicitly selected numerical backend is unavailable."""


class BackendExecutionError(RuntimeError):
    """The selected backend failed or returned inconsistent/nonfinite output."""


@dataclass
class BioequivalenceResult:
    # Legacy attributes remain for callers. New exports intentionally omit names
    # that previously implied confidence, bootstrap inference or FDA acceptance.
    f2: float
    conclusion: str
    r_backend_used: bool
    ci_low: float | None = None
    ci_high: float | None = None
    bootstrap_median: float | None = None
    bootstrap_p05: float | None = None
    bootstrap_p95: float | None = None
    probability_f2_ge_50: float | None = None
    bootstrap_runs: int = 0
    cv_flag: str = "Not assessed"
    method_note: str = ""
    fda_decision: str = "Not established"
    fda_risk: str = "Review needed"
    fda_next_action: str = "Qualified product-specific review required."
    schema: str = RESULT_SCHEMA
    f2_unrounded: float | None = None
    descriptive_threshold_met: bool = False
    backend: str = "python"
    backend_version: str = ""
    seed: int = 1729
    simulation_runs: int = 0
    simulation_interval_low: float | None = None
    simulation_interval_high: float | None = None
    simulation_median: float | None = None
    simulation_p05: float | None = None
    simulation_p95: float | None = None
    simulated_fraction_f2_ge_50: float | None = None
    simulation_assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    applicability_issues: list[str] = field(default_factory=list)
    applicability_status: str = "Review needed"
    variability_policy: str = VARIABILITY_POLICY
    profile_fingerprint: str = ""
    time_points_used: list[float] = field(default_factory=list)
    clinical_be_established: bool = False
    regulatory_acceptance: str = "Not established"

    def to_dict(self):
        result = asdict(self)
        for name in ("ci_low", "ci_high", "bootstrap_median", "bootstrap_p05", "bootstrap_p95", "probability_f2_ge_50", "bootstrap_runs", "fda_decision", "fda_risk", "fda_next_action"):
            result.pop(name)
        return result


def _clean_profile(profile_df):
    """Validate every supplied row; normalize ordering without dropping points."""
    if not isinstance(profile_df, pd.DataFrame):
        raise ProfileValidationError("Profile must be a pandas DataFrame with all seven required columns.")
    if profile_df.columns.duplicated().any():
        raise ProfileValidationError("Duplicate column names are not supported.")
    missing = [column for column in REQUIRED_COLUMNS if column not in profile_df.columns]
    if missing:
        raise ProfileValidationError("Missing required column(s): " + ", ".join(missing))
    if len(profile_df) < 3:
        raise ProfileValidationError("At least three common nonzero dissolution time points are required.")
    df = profile_df[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        if df[column].map(lambda value: isinstance(value, (complex, np.complexfloating))).any():
            raise ProfileValidationError(f"{column} must contain real numeric values; complex components are not discarded.")
        boolean_rows = df[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        values = pd.to_numeric(df[column], errors="coerce")
        try:
            valid = np.isfinite(values.to_numpy(dtype=float)) & ~boolean_rows.to_numpy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProfileValidationError(f"{column} must contain finite numeric values.") from exc
        if not valid.all():
            rows = ", ".join(str(index + 1) for index in np.flatnonzero(~valid))
            raise ProfileValidationError(f"{column}: finite numeric values required at row(s) {rows}; no rows were dropped or filled.")
        df[column] = values.astype(float)
    if (df["Time (min)"] <= 0).any():
        raise ProfileValidationError("Time points must be positive; exclude time zero explicitly under the study protocol.")
    if df["Time (min)"].duplicated().any():
        raise ProfileValidationError("Duplicate time points would change f2 weighting; supply unique common time points.")
    for arm in ("Reference", "Test"):
        means, sd, counts = (df[f"{arm} {suffix}"] for suffix in ("Mean (%)", "SD", "n"))
        if ((means < 0) | (means > 100)).any():
            raise ProfileValidationError(f"{arm} means must be within this model's supported 0–100% range. Analytical over-recovery requires separate review; values are not clipped.")
        if (sd < 0).any():
            raise ProfileValidationError(f"{arm} sample SD must be nonnegative; negative values are not clamped.")
        if ((counts < 2) | (counts > 2147483647) | (counts != np.floor(counts))).any():
            raise ProfileValidationError(f"{arm} n must be an integer from 2 through 2147483647; sample SD requires at least two units.")
        if counts.nunique() != 1:
            raise ProfileValidationError(f"{arm} n must be constant across time points for this summary-profile model. Missing units require study-level review.")
    return df.sort_values("Time (min)", kind="stable").reset_index(drop=True)


def _f2_values(ref, test, axis=-1):
    # Stable RMS/log form avoids squaring large simulation draws directly.
    difference = np.asarray(ref, dtype=float) - np.asarray(test, dtype=float)
    if not np.isfinite(difference).all():
        raise BackendExecutionError("Nonfinite profile differences; review SDs and simulation assumptions.")
    scale = np.max(np.abs(difference), axis=axis, keepdims=True)
    normalized = np.divide(difference, scale, out=np.zeros_like(difference), where=scale != 0)
    rms = np.squeeze(scale, axis=axis) * np.sqrt(np.mean(normalized ** 2, axis=axis))
    return 50 * (2 - np.log10(np.hypot(1.0, rms)))


def _f2_from_means(ref_values, test_values):
    if np.iscomplexobj(ref_values) or np.iscomplexobj(test_values):
        raise ProfileValidationError("f2 requires real profile means; complex components are not discarded.")
    ref, test = np.asarray(ref_values, dtype=float), np.asarray(test_values, dtype=float)
    if ref.ndim != 1 or test.ndim != 1 or ref.shape != test.shape or ref.size < 3:
        raise ProfileValidationError("f2 requires equal one-dimensional profiles with at least three time points.")
    if not np.isfinite(ref).all() or not np.isfinite(test).all():
        raise ProfileValidationError("f2 profile means must be finite.")
    return float(_f2_values(ref, test))


def _cv_issues(df):
    issues = []
    for position, row in df.iterrows():
        threshold = 20 if position == 0 else 10
        for arm in ("Reference", "Test"):
            mean, sd = row[f"{arm} Mean (%)"], row[f"{arm} SD"]
            if mean == 0:
                issues.append(f"{arm} CV undefined at {row['Time (min)']:g} min because mean is zero.")
            elif sd / mean * 100 > threshold:
                issues.append(f"{arm} CV {sd / mean * 100:.2f}% at {row['Time (min)']:g} min exceeds the named {threshold}% screening threshold.")
    return issues


def _cv_flag(df):
    issues = _cv_issues(df)
    return "; ".join(issues) if issues else "No exceedance in the named CV screen; eligibility not established"


def _bootstrap_stats(values):
    """Legacy helper name: percentiles of supplied simulated f2 values only."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise BackendExecutionError("Simulation must return a nonempty finite vector of f2 values.")
    low, p05, median, p95, high = np.percentile(arr, [2.5, 5, 50, 95, 97.5], method="linear")
    fraction = float(np.mean(arr >= 50))
    return {"ci_low": float(low), "ci_high": float(high), "median": float(median), "p05": float(p05), "p95": float(p95),
            "probability_f2_ge_50": fraction * 100, "simulated_fraction_f2_ge_50": fraction}


def _simulation_settings(bootstrap_runs, seed, simulation_runs=_UNSET):
    for value in (bootstrap_runs, simulation_runs):
        if value is not _UNSET and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 100000):
            raise ProfileValidationError("Simulation runs must be an integer from 1 through 100000.")
    if simulation_runs is not _UNSET and bootstrap_runs is not _UNSET and simulation_runs != bootstrap_runs:
        raise ProfileValidationError("Specify simulation_runs or legacy bootstrap_runs; conflicting counts are not supported.")
    runs = simulation_runs if simulation_runs is not _UNSET else bootstrap_runs if bootstrap_runs is not _UNSET else 1000
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2147483647:
        raise ProfileValidationError("Seed must be an integer from 0 through 2147483647 for the supported Python/R backends.")
    return int(runs), int(seed)


def _python_bootstrap_stats(df, bootstrap_runs=1000, seed=1729):
    """Legacy helper name: independent-normal sample-mean simulation, not bootstrap."""
    runs, seed = _simulation_settings(bootstrap_runs, seed)
    means = df[["Reference Mean (%)", "Test Mean (%)"]].to_numpy(dtype=float).reshape(-1)
    scales = np.column_stack((df["Reference SD"] / np.sqrt(df["Reference n"]),
                              df["Test SD"] / np.sqrt(df["Test n"]))).reshape(-1)
    rng = np.random.Generator(np.random.PCG64(seed))
    with np.errstate(over="ignore", invalid="ignore"):
        draws = rng.normal(means, scales, size=(runs, means.size))
    if not np.isfinite(draws).all():
        raise BackendExecutionError("Normal simulation produced nonfinite draws; review SDs and model assumptions. No draws were dropped.")
    return _bootstrap_stats(_f2_values(draws[:, 0::2], draws[:, 1::2], axis=1))


def _run_r_backend(df, bootstrap_runs=1000, seed=1729):
    executable = shutil.which("Rscript")
    if not executable:
        raise BackendUnavailableError("R was selected but Rscript is unavailable. Select Python explicitly or configure R; no fallback was performed.")
    r_code = r"""
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], check.names = FALSE)
runs <- as.integer(args[3]); seed <- as.integer(args[4])
f2_calc <- function(ref, test) {
  delta <- ref-test; scale <- max(abs(delta))
  rms <- if (scale == 0) 0 else scale * sqrt(mean((delta/scale)^2))
  log_norm <- if (rms > 1) log10(rms) + 0.5*log10(1+(1/rms)^2) else 0.5*log10(1+rms^2)
  50 * (2 - log_norm)
}
RNGkind("Mersenne-Twister", "Inversion"); set.seed(seed)
mu <- as.vector(t(as.matrix(d[c("Reference Mean (%)", "Test Mean (%)")])))
se <- as.vector(t(cbind(d[["Reference SD"]]/sqrt(d[["Reference n"]]), d[["Test SD"]]/sqrt(d[["Test n"]]))))
values <- replicate(runs, {
  draws <- rnorm(length(mu), mean=mu, sd=se)
  f2_calc(draws[seq(1,length(mu),by=2)], draws[seq(2,length(mu),by=2)])
})
if (!all(is.finite(values))) stop("Simulation produced nonfinite values; no draws were dropped")
q <- quantile(values, c(.025,.05,.5,.95,.975), type=7, names=FALSE)
out <- data.frame(f2=f2_calc(d[["Reference Mean (%)"]],d[["Test Mean (%)"]]),
 ci_low=q[1],p05=q[2],median=q[3],p95=q[4],ci_high=q[5],
 probability_f2_ge_50=mean(values>=50)*100,backend_version=paste(R.version.string,"Mersenne-Twister/Inversion;quantile-type7"))
write.csv(out,args[2],row.names=FALSE)
"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path, script_path = [os.path.join(tmpdir, name) for name in ("profile.csv", "result.csv", "f2_simulation.R")]
            df.to_csv(input_path, index=False, quoting=csv.QUOTE_MINIMAL)
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(r_code)
            subprocess.run([executable, script_path, input_path, output_path, str(bootstrap_runs), str(seed)],
                           check=True, capture_output=True, text=True, timeout=30)
            table = pd.read_csv(output_path)
            if len(table) != 1:
                raise ValueError("R returned an unexpected result row count")
            result = table.iloc[0]
            return {key: float(result[key]) for key in ("f2", "ci_low", "ci_high", "median", "p05", "p95", "probability_f2_ge_50")} | {"backend_version": str(result["backend_version"])}
    except BackendUnavailableError:
        raise
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, pd.errors.ParserError) as exc:
        detail = (getattr(exc, "stderr", "") or str(exc)).strip()[:500]
        raise BackendExecutionError(f"Selected R backend failed: {detail}. No Python fallback was performed.") from exc


def _validate_stats(stats, expected_f2=None):
    try:
        ordered = [float(stats[name]) for name in ("ci_low", "p05", "median", "p95", "ci_high")]
        percent = float(stats["probability_f2_ge_50"])
        if not all(math.isfinite(value) for value in ordered + [percent]) or ordered != sorted(ordered) or not 0 <= percent <= 100 or max(ordered) > 100 + 1e-8:
            raise ValueError("nonfinite, unordered or out-of-range simulation statistics")
        if expected_f2 is not None and not math.isclose(float(stats["f2"]), expected_f2, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("R point-f2 disagrees with the checked descriptive calculation")
        version = stats.get("backend_version")
        if expected_f2 is not None and (not isinstance(version, str) or version.strip().casefold() in {"", "none", "nan"}):
            raise ValueError("R did not report its version")
    except (TypeError, ValueError, KeyError) as exc:
        raise BackendExecutionError(f"Selected backend returned invalid statistics: {exc}. No fallback was performed.") from exc


def _profile_fingerprint(df):
    payload = {"columns": REQUIRED_COLUMNS, "values": df.to_numpy(dtype=float).tolist()}
    return hashlib.sha256(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def calculate_f2(profile_df, bootstrap_runs=_UNSET, seed=1729, *, backend="python", simulation_runs=_UNSET):
    """Descriptive f2 and disclosed independent-normal summary-data simulation.

    bootstrap_runs is a compatibility alias for simulation_runs, not an empirical
    bootstrap request. Backend selection is explicit and never falls back.
    """
    runs, seed = _simulation_settings(bootstrap_runs, seed, simulation_runs)
    if backend not in {"python", "r"}:
        raise ProfileValidationError("Backend must be explicitly 'python' or 'r'.")
    df = _clean_profile(profile_df)
    if runs * len(df) * 2 > 5000000:
        raise ProfileValidationError("Requested simulation exceeds the 5,000,000-cell computation limit. Reduce iterations; no input points were removed.")
    point = _f2_from_means(df["Reference Mean (%)"], df["Test Mean (%)"])
    threshold_met = point >= 50
    if backend == "r":
        stats = _run_r_backend(df, bootstrap_runs=runs, seed=seed)
        _validate_stats(stats, expected_f2=point)
        backend_version = stats["backend_version"]
    else:
        stats = _python_bootstrap_stats(df, bootstrap_runs=runs, seed=seed)
        _validate_stats(stats)
        backend_version = f"Python {platform.python_version()}; NumPy {np.__version__}; PCG64; linear percentiles"
    issues = _cv_issues(df)
    for arm in ("Reference", "Test"):
        if df[f"{arm} n"].iloc[0] < 12:
            issues.append(f"{arm} n is below 12 units; descriptive computation does not establish conventional f2 applicability.")
        if (df[f"{arm} Mean (%)"] > 85).sum() > 1:
            issues.append(f"{arm} includes more than one mean above 85%; protocol-based tail-point selection needs review. All supplied points were retained.")
        if (df[f"{arm} Mean (%)"].diff().dropna() < 0).any():
            issues.append(f"{arm} mean profile is nonmonotonic (decreases); review cumulative dissolution data and analytical assumptions.")
    issues.append("Study protocol, media/method, raw unit profiles and product-specific eligibility have not been verified.")
    if runs < 1000:
        issues.append("Fewer than 1000 simulation runs; tail summaries can have substantial Monte Carlo variability. This is not an external validation threshold.")
    conclusion = ("Descriptive f2 is at least 50" if threshold_met else "Descriptive f2 is below 50") + "; clinical BE and regulatory acceptance are not established."
    low, high = float(stats["ci_low"]), float(stats["ci_high"])
    return BioequivalenceResult(
        f2=round(point, 2), f2_unrounded=point, descriptive_threshold_met=threshold_met,
        conclusion=conclusion, r_backend_used=backend == "r", backend=backend, backend_version=backend_version, seed=seed,
        ci_low=round(low, 2), ci_high=round(high, 2), bootstrap_median=round(stats["median"], 2),
        bootstrap_p05=round(stats["p05"], 2), bootstrap_p95=round(stats["p95"], 2),
        probability_f2_ge_50=round(stats["probability_f2_ge_50"], 1), bootstrap_runs=runs,
        simulation_runs=runs, simulation_interval_low=low, simulation_interval_high=high,
        simulation_median=float(stats["median"]), simulation_p05=float(stats["p05"]), simulation_p95=float(stats["p95"]),
        simulated_fraction_f2_ge_50=float(stats["probability_f2_ge_50"]) / 100,
        cv_flag=_cv_flag(df), simulation_assumptions=list(SIMULATION_ASSUMPTIONS), limitations=list(LIMITATIONS),
        applicability_issues=issues, profile_fingerprint=_profile_fingerprint(df), time_points_used=df["Time (min)"].tolist(),
        method_note=f"Independent-normal sample-mean simulation using explicitly selected {backend} backend; 2.5th/97.5th simulation percentiles, not calibrated confidence limits. No fallback performed.",
        fda_decision="Not established; descriptive dissolution output only", fda_risk="Review needed",
        fda_next_action="Qualified review of the prespecified protocol, methods, unit-level data and applicable product-specific BE or biowaiver evidence is required.",
    )


def dissolution_profile_summary(profile_df):
    df = _clean_profile(profile_df)
    df["Difference (%)"] = df["Test Mean (%)"] - df["Reference Mean (%)"]
    for arm in ("Reference", "Test"):
        df[f"{arm} CV (%)"] = np.divide(df[f"{arm} SD"] * 100, df[f"{arm} Mean (%)"],
            out=np.full(len(df), np.nan), where=df[f"{arm} Mean (%)"].to_numpy() > 0)
    return df.round(2)
