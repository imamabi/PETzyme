# PETzyme

PETzyme is a research assistant for exploring experimental evidence about enzymes
that can act on polyethylene terephthalate (PET). It helps a researcher inspect
recorded conditions and responses, identify evidence gaps, and consider a
conditional next experiment. It does not establish an enzyme's superiority,
predict PET activity, or supply a validated laboratory protocol.

**Ask PETzyme** is a local Streamlit interface. When configured on macOS, the
Meta Muse model can select scientific tools and organize their grounded results;
the tools and a labelled deterministic offline path also work without an LLM.
Rather than only retrieving passages, the workflow executes scientific tools,
checks evidence provenance and scope, and grounds accepted results in those
executions; it still cannot establish missing scientific facts.
The public product name is PETzyme; the Python package remains `zymecycle` for
compatibility.

## What it does

- Retrieves original NREL observations and bounded, publication-derived PAZy
  activity evidence, retaining source, assay conditions and individual values.
- Assesses evidence, missing conditions and experimental comparability before
  suggesting hypothetical next actions. Source overlap is not independent
  replication, and matching names do not establish construct identity.
- Triages protein names or user-supplied sequences and can compute local ESM-2
  representation similarities. Similarity is **not** measured PET activity.
- Uses canonical facts from executed tools, source-specific provenance and
  explicit `MEASURED`, `COMPUTED` and `HYPOTHETICAL` distinctions. A successful
  tool call does not guarantee a scientifically complete answer.

Example questions:

- “Investigate ESM050. What experimental conditions and outcomes are reported,
  and what should I test next?”
- “Show NREL evidence for ESM050 at 60°C.”
- “What are five enzymes with high recorded PET responses in a comparable assay
  group?”

Results are limited to the data and scope actually retrieved. The source unit of
the condensed NREL CSV is not explicitly declared: original values are retained,
and eligible ×1,000 display values are labelled as **inferred and computed**.
PAZy values retain their source-reported units. Some historical comparisons and
record-level cross-source identity remain unresolved. The interface must not be
used as evidence of statistical significance, causality, an established optimum,
or a prediction of experimental activity.

## Local setup

Python 3.11 or later is required. The project is run from its repository root;
no REST API or React frontend is included. A fresh checkout does **not** include
the scientific data or model weights needed for all features.

```sh
python3.11 -m venv .venv
# Apple Silicon / Python 3.11: the dependency set used for release verification.
.venv/bin/python -m pip install -r requirements-macos-py311.lock
```

The version lock records the isolated test environment; it does not pin wheel
hashes or guarantee other platforms. For a lighter UI-only install, or to resolve
dependencies on another supported platform, use the pinned top-level packages:

```sh
.venv/bin/python -m pip install 'httpx==0.28.1' 'streamlit==1.66.1' 'pytest==9.1.1'
# Needed for ESM-2 and complete test collection:
.venv/bin/python -m pip install 'torch==2.8.0' 'transformers==4.57.6'
```

Real ESM-2 inference also requires the pinned model files described in
[MODEL_PROVENANCE.json](MODEL_PROVENANCE.json). Some tests import PyTorch even
when the real-model smoke test is disabled.

### Local data and model assets

The following files are excluded from Git and must be provisioned separately:

- **NREL:** `data/nrel/label_data.csv` from the archive in the
  [Zenodo record](https://zenodo.org/records/15417757), with the checksum and
  attribution in [NREL_SOURCE_MANIFEST.json](NREL_SOURCE_MANIFEST.json). Most
  NREL and combined investigations require this file. Check its SHA-256 against
  `unit_reconciliation.csv_sha256` in the manifest before use.
- **PAZy activity:** the one checksum-pinned archive at the `local_archive` path
  in [PAZY_SOURCE_MANIFEST.json](PAZY_SOURCE_MANIFEST.json). Combined activity
  retrieval and discovery require it. Review its source terms before obtaining
  or using it; the manifest records unresolved reuse licensing. Do not distribute
  the archive or derived datasets with this repository.
- **ESM-2:** the pinned checkpoint files under `models/`, as specified by the
  model manifest. They are needed only for ESM-2 analysis; inference verifies
  local hashes and does not download the model at runtime.

After reviewing the PAZy source terms for your intended use, prepare the
required resources from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_nrel
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_pazy_activity
# Optional: downloads the pinned local ESM-2 checkpoint and tokenizer.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_esm2
```

These commands use the official sources and verify pinned integrity. NREL setup
preserves an existing reconciliation manifest and reuses matching local data;
it fails on a checksum mismatch. PAZy setup checks the pinned archive checksum,
schema and counts without printing records. The model setup preserves an
existing populated model manifest. All three assets stay in Git-ignored local
directories; do not redistribute the PAZy archive. Setup requires network
access; the applications do not download scientific assets automatically.

### Run

For local, deterministic operation with no paid Muse calls:

```sh
env PETZYME_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m streamlit run streamlit_app.py
```

Open <http://127.0.0.1:8501/>. Omit `PETZYME_OFFLINE=1` to use Muse when the
existing macOS login-Keychain item is available. The item uses service
`zymecycle-meta-api` and account `zymecycle`; enter the credential directly in
Keychain Access, never in source, a shell command, `.env`, logs or chat. The
launcher supplies `MODEL_API_KEY` to the child process only. If Muse is
unavailable, the UI can show a clearly labelled deterministic local result.
Live inference uses external Meta services and may incur usage charges.
The one-shot `scripts.research_agent` CLI can also read `MODEL_API_KEY` from its
own environment when explicitly authorized; use an appropriate secret manager
for your operating system. The Streamlit launcher itself requires macOS Keychain
for live Muse and does not offer a portable cloud-secret integration.

## Tests and interfaces

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
# Explicit opt-in: requires the verified local checkpoint and optional packages.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  ZYMECYCLE_REAL_ESM=1 .venv/bin/python -m pytest -q
```

The complete integration suite also expects the pinned local scientific data;
without those assets, a fresh checkout is not expected to pass every test. The
existing tool catalog is available through `zymecycle.tools.describe_tools()`
and `invoke_tool()`. The optional one-shot CLI is
`python -m scripts.research_agent`; see its `--help` for its separate live-call
and runtime controls.

## Sources and privacy

- NREL dataset: Norton-Baker et al., [DOI 10.5281/zenodo.15417757](https://doi.org/10.5281/zenodo.15417757);
  the source manifest records CC BY 4.0 attribution. The related study is
  [DOI 10.1021/acscatal.5c03460](https://doi.org/10.1021/acscatal.5c03460).
  Its supporting workbook, used for reconciliation, is recorded separately as
  CC BY-NC 4.0; the applicability of those terms to the derived 1,570-cell
  certificate in the manifest requires review before public redistribution.
- [PAZy](https://www.pazy.eu/) and Feuerriegel et al.,
  [DOI 10.1093/nar/gkag1026](https://doi.org/10.1093/nar/gkag1026).
  The activity archive is publication-derived curation, not new independent
  experiments. PAZy protein metadata and sequences are retrieved live only;
  API responses, protein records and restricted activity projections must not be
  persisted, exported or logged.
- ESM-2: `facebook/esm2_t12_35M_UR50D`; checkpoint, revision, hashes and model
  license are recorded in [MODEL_PROVENANCE.json](MODEL_PROVENANCE.json).

NREL and PAZy source identities, measurement definitions and attribution remain
separate. PANDA is not used. Secrets, datasets, weights, user sequences, provider
transcripts and generated outputs do not belong in Git. Publication and reuse
rights for the PAZy archive and any source-derived supporting material require
review; open API access alone is not a redistribution license.

## Project history and layout

The private development history records incremental scientific foundations,
agent work and interface builds beginning on 27 October 2026. It is not, by
itself, a verified attribution of which work preceded the hackathon; the owner
should identify pre-existing foundations and event-period contributions in any
submission. The candidate is a fresh snapshot, not a copy of private history.

- `zymecycle/` — scientific instruments, contracts, validation and agent logic.
- `streamlit_app.py` and `zymecycle/ui_*.py` — researcher interface and adapters.
- `scripts/` — setup, secure launcher, CLI and historical demonstrations.
- `tests/` — scientific, agent, privacy and interface regression tests.
- `*_MANIFEST.json` and `MODEL_PROVENANCE.json` — source and model provenance.

The PETzyme project's own code license has not yet been specified. Third-party
data, publications and model weights have their own terms; do not assume those
terms license the project code or authorize redistribution of their contents.
