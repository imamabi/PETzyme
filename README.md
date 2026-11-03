## How PETzyme Works

PETzyme treats an enzyme investigation as a **scientific workflow**, not simply a question-answering task.

Meta Muse helps interpret research objectives and select scientific tools. Python instruments execute the investigation against structured experimental records, while independent validation checks determine which measurements, comparisons, and conclusions are supported.

```mermaid
flowchart TB
    A["Research Question<br/>Enzyme · Conditions · Objective"]
    B["Meta Muse Agent<br/>Interpret & Select Scientific Tools"]
    C["Execute Scientific Instruments<br/>NREL · PAZy · ESM-2 when relevant"]
    D["Structured Experimental Evidence<br/>Measurements · Conditions · Source Records"]
    E["Scientific Validation<br/>Units · Identity · Comparability · Grounding"]
    F{"What does the evidence support?"}
    G["Supported Findings<br/>Measured Values · Qualified Computations"]
    H["Evidence Gap or Incompatible Assays<br/>No Unsupported Comparison"]
    I["Conditional Experiment Planning<br/>Laboratory Constraints · Next Action"]
    J["Researcher Follow-up<br/>Refine Conditions or Investigate Further"]

    A --> B --> C --> D --> E --> F
    F -->|Supported| G
    F -->|Unsupported| H
    G -.->|When planning is requested| I
    G --> J
    H --> J
    I --> J
    J --> A

    classDef input fill:#F1F5F9,stroke:#64748B,color:#0F172A
    classDef agent fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef science fill:#D1FAE5,stroke:#059669,color:#064E3B
    classDef validation fill:#FEF3C7,stroke:#D97706,color:#78350F
    classDef output fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95

    class A,J input
    class B agent
    class C,D science
    class E,F validation
    class G,H,I output
```

A deterministic offline route can also execute the scientific workflow without Muse.

### An example: investigating ESM050

In a source-backed investigation, PETzyme identified 16 retrieved PAZy assay records associated with the same NREL publication. Rather than treating their presence across databases as independent replication, PETzyme identifies the shared publication lineage.

It also distinguishes original NREL measurements from values converted using an inferred unit, preserves individual PAZy observations, and qualifies comparisons when experimental equivalence is unverified.

The result is more than a retrieved answer: **an auditable scientific assessment that can inform a conditional next experiment.**
