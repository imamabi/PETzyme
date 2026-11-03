# PETzyme

**An AI research assistant for PET-degrading enzyme discovery and evidence-grounded experimental planning.**

Finding promising enzymes for plastic recycling requires more than comparing reported activity values. Experimental measurements come from different studies, substrates, temperatures, and protocols. Without that context, comparisons can be misleading.

**PETzyme** brings these measurements into an interactive research workflow. Powered by **Meta Muse Spark 1.3** and deterministic Python scientific tools, it helps researchers investigate PET-degrading enzymes, evaluate experimental evidence, and identify meaningful next experiments.

Unlike a conventional retrieval-based chatbot, PETzyme works with structured experimental records. Muse supports scientific tool orchestration, while Python handles evidence retrieval, calculations, provenance checks, and scientific validation.

**The goal: turn scattered experimental measurements into research decisions that scientists can inspect, question, and test.**

## How PETzyme Works

```mermaid
flowchart LR
    A["Research Question"] --> B["Meta Muse<br/>Tool Orchestration"]
    B --> C["Scientific Tools<br/>NREL · PAZy · ESM-2<br/>Analysis & Validation"]
    C --> D["Grounded Findings<br/>& Next Experiments"]

    classDef input fill:#F1F5F9,stroke:#64748B,color:#0F172A
    classDef agent fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A
    classDef science fill:#D1FAE5,stroke:#059669,color:#064E3B
    classDef output fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95

    class A input
    class B agent
    class C science
    class D output
```

Meta Muse helps interpret research questions and select scientific tools. The Python engine retrieves experimental measurements, evaluates their scientific context, and grounds the resulting findings in executed evidence.

PETzyme can also run a deterministic offline workflow without an LLM. Experimental recommendations remain conditional on the available evidence and laboratory constraints.

## What PETzyme Can Do

- **Investigate enzymes** — retrieve NREL and PAZy experimental measurements with their assay conditions, responses, units, and source references.
- **Evaluate evidence** — identify missing conditions, assess experimental comparability, and distinguish source overlap from independent replication.
- **Explore protein sequences** — investigate enzyme identities and calculate local ESM-2 representation similarities. Similarity is not measured PET-degrading activity.
- **Plan experiments** — propose conditional next experiments based on recorded evidence and researcher-supplied laboratory constraints.
- **Continue investigations** — ask follow-up questions, change evidence sources, and refine experimental conditions within a research conversation.

### An example: investigating ESM050

In a source-backed ESM050 investigation, PETzyme identified **16 retrieved PAZy assay records associated with the same underlying NREL publication**.

Although the measurements appeared across different data resources, they did not establish independent experimental replication.

PETzyme also distinguishes original measurements from inferred unit conversions and flags comparisons that lack sufficient experimental equivalence.

This is the value of going beyond retrieval: **the system helps researchers understand what the evidence actually supports.**

## Try Asking PETzyme

> Investigate ESM050. What experimental results have been reported, and what should I test next?

> Show NREL evidence for ESM050 at 60°C.

> Which five enzymes have high recorded PET-degradation responses under comparable assay conditions?

> My laboratory supports only 40°C and pH 7 or lower. What experiment should I consider next?

## Scientific Grounding

PETzyme separates experimental observations, computational results, and proposed experiments.

| Evidence class | Meaning |
|---|---|
| **MEASURED** | Experimental observations retrieved with their source and assay conditions |
| **COMPUTED** | Values calculated by scientific tools, with assumptions and inferred units explicitly identified |
| **HYPOTHETICAL** | Conditional experimental recommendations, not validated protocols or predicted outcomes |

Scientific safeguards help PETzyme:

- Reject unsupported comparisons across incompatible assays.
- Distinguish genuine zero responses from missing measurements.
- Identify repeated evidence originating from the same publication.
- Preserve source-specific units and provenance.
- Avoid treating similar enzyme names as proof of identical constructs.
- Respect experimental constraints and acknowledge missing information.

**Known limitations:** The condensed NREL CSV does not explicitly declare its response unit. Eligible converted values are therefore labelled *inferred/computed*, while the original observations are preserved. PAZy values retain their source-reported units. Cross-source enzyme identity and experimental comparability remain qualified where sufficient evidence is unavailable.

PETzyme does not establish enzyme superiority, predict PET-degradation activity, prove causality, or provide experimentally validated laboratory protocols.

## Getting Started

### Requirements

- Python 3.11
- macOS with Apple Silicon for the verified dependency environment
- Internet access for initial scientific resource preparation
- Meta Muse credentials for optional live agent execution

The scientific tools also support deterministic offline operation without Muse.

### 1. Install dependencies

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-py311.lock
```

The dependency lock reflects the verified release environment. Other platforms may require separate dependency resolution.

### 2. Prepare scientific resources

The repository does not bundle scientific datasets or pretrained model weights. The preparation scripts acquire resources from their official sources and verify their integrity.

```bash
# NREL experimental measurements
.venv/bin/python -m scripts.prepare_nrel

# PAZy experimental activity data
.venv/bin/python -m scripts.prepare_pazy_activity

# Optional: local ESM-2 model
.venv/bin/python -m scripts.prepare_esm2
```

Resource versions, checksums, and provenance are documented in:

- `NREL_SOURCE_MANIFEST.json`
- `PAZY_SOURCE_MANIFEST.json`
- `MODEL_PROVENANCE.json`

Scientific assets are stored in Git-ignored local directories. Initial preparation requires network access; scientific assets are not downloaded automatically during an investigation.

### 3. Launch PETzyme

For deterministic offline operation without paid Muse inference:

```bash
PETZYME_OFFLINE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m streamlit run streamlit_app.py
```

Open **http://127.0.0.1:8501** in your browser.

Omit `PETZYME_OFFLINE=1` to allow Muse when a
credential is available. On macOS, the existing login-Keychain item takes
precedence: enter it directly in Keychain Access using service
`zymecycle-meta-api` and account `zymecycle`. If no valid Keychain credential
is available, the launcher uses `MODEL_API_KEY` from its process environment.
The same environment route is available on Linux and Windows. Never enter the
key in PETzyme's question box or save it in source, `.env`, logs or chat.

For macOS or Linux, a temporary shell session can read the key without echoing
it (Bash; do not put the actual key in the command or shell history):

```sh
read -r -s MODEL_API_KEY; export MODEL_API_KEY; printf '\n'
.venv/bin/python -m streamlit run streamlit_app.py
unset MODEL_API_KEY
```

For Windows PowerShell, the following reads the key with masked input and keeps
it in the current process environment until the session ends:

```powershell
$secure = Read-Host 'Meta API key' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $env:MODEL_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr); Remove-Variable secure,ptr }
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
Remove-Item Env:MODEL_API_KEY
```

Environment variables are inherited by child processes and may be visible to
other processes with sufficient access; use an appropriate secret manager and
avoid shared or untrusted environments. The key is forwarded to the existing
local worker, and that worker sends it only to the configured Meta API. Without
a usable key, the UI provides labelled deterministic local results. Live
inference uses external Meta services and may incur usage charges.

macOS on Apple Silicon with Python 3.11 has been exercised locally. Linux and
Windows credential and worker branches have automated simulations, but native
installation, startup and model inference on those platforms have not yet
been verified. Install the existing pinned dependencies in a Python 3.11+
virtual environment using platform-compatible packages; platform-specific
PyTorch availability and local data/model setup may require separate review.
For Windows PowerShell, the corresponding environment setup is:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install 'httpx==0.28.1' 'streamlit==1.66.1' 'pytest==9.1.1'
# For the ESM-2 capability and full test collection:
.venv\Scripts\python.exe -m pip install 'torch==2.8.0' 'transformers==4.57.6'
```

The one-shot `scripts.research_agent` CLI can also read `MODEL_API_KEY` from its
own environment when explicitly authorized; use an appropriate secret manager
for your operating system.

## Testing

Run the scientific and integration tests:

```bash
.venv/bin/python -m pytest -q
```

To include real local ESM-2 inference:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  ZYMECYCLE_REAL_ESM=1 \
  .venv/bin/python -m pytest -q
```

The complete suite requires the prepared scientific datasets and ESM-2 checkpoint.

**Release verification: 1,310 tests passed, zero skipped, including real ESM-2 inference in an isolated environment.**

These tests validate software behavior and scientific safeguards; they do not constitute laboratory validation of proposed experiments.

## Project Structure

| Component | Purpose |
|---|---|
| `streamlit_app.py` | Researcher-facing interface |
| `petzyme/` | Scientific tools, agent orchestration, validation, and UI adapters |
| `scripts/` | Resource preparation, launch utilities, and CLI workflows |
| `tests/` | Scientific, agent, privacy, and integration tests |
| `*_MANIFEST.json` | Experimental source provenance and reconciliation |
| `MODEL_PROVENANCE.json` | ESM-2 checkpoint and model integrity |

**PETzyme** is the public project name. The Python package is `petzyme`.

## Scientific Sources

PETzyme builds on published research resources:

- **NREL:** Norton-Baker et al., *ACS Catalysis* (2025). [Research article](https://doi.org/10.1021/acscatal.5c03460) · [Experimental dataset](https://doi.org/10.5281/zenodo.15417757)
- **PAZy:** Feuerriegel et al., *Nucleic Acids Research* (2026). [Publication](https://doi.org/10.1093/nar/gkag1026) · [PAZy database](https://www.pazy.eu/)
- **ESM-2:** Meta's pretrained protein language model, [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

Resource versions, integrity checks, source attribution, and relevant usage qualifications are documented in the accompanying provenance manifests.

---

**PETzyme — From experimental evidence to the next research question.**
