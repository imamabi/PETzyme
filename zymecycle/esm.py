"""Pinned, local-only ESM-2 backbone inference and representation comparison."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import Source, ToolResult
from .identity import sequence_fingerprint
from .protein_sequences import MAX_RESIDUES, CANONICAL
from .protein_types import (EmbeddingComparison, EmbeddingRuntime, ModelSpec,
                            PreparedSequence, ProteinEmbedding)
from .results import finite_number, outcome

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    'ESM-2 cosine similarity is a computed sequence-representation metric, not measured PET activity.',
    'Similarity establishes no catalytic efficiency, thermal stability, structural identity or functional equivalence.',
    'Biological superiority is UNKNOWN; no active/promising thresholds, predictions or mutation recommendations are made.',
    'Mean pooling discards residue order in the final representation; token representations remain context dependent.',
    'Float32 inference may differ slightly by device/library version; reproducibility is checked within tolerance.',
]


def model_source(spec: ModelSpec) -> Source:
    """Build the ESM-2 model Source (checkpoint URL, paper DOI, layer/pooling locator).

    Flow: Receives ModelSpec from embed_protein(); returns Source for its model-attributed ToolResult and find_similar_nrel_proteins()."""
    return {'name': 'ESM-2', 'url': f"https://huggingface.co/{spec['checkpoint']}/tree/{spec['revision']}",
            'doi': '10.1126/science.ade2574', 'locator': f"layer={spec['layer']}; pooling={spec['pooling']}",
            'attribution': 'Meta AI / FAIR, ESM-2, Lin et al. (2023); checkpoint license ' + spec['license'],
            'retrieved_at_utc': None}


def pool_residue_embeddings(hidden: Tensor, attention_mask: Tensor, special_tokens_mask: Tensor,
                            expected_residues: int) -> list[float]:
    """Mean-pool residue (non-special, non-padded) token vectors into one list[float].

    Purpose: single fixed-size representation per protein. Refuses shape
        mismatches, residue-count mismatches, and non-finite values
        instead of producing a silently wrong vector.
    Raises:
        ValueError: on invalid tensor/mask shapes or non-finite values.
    Flow: Receives backbone Tensor and tokenizer masks from ESM2Runtime.encode(); returns mean-pooled list[float] for embed_protein()."""
    import torch
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or attention_mask.shape != hidden.shape[:2]
            or special_tokens_mask.shape != attention_mask.shape):
        raise ValueError('Invalid residue tensor/mask shapes')
    valid = attention_mask.bool() & ~special_tokens_mask.bool()
    if int(valid.sum().item()) != expected_residues or expected_residues <= 0:
        raise ValueError('Residue count mismatch; no truncation or unknown tokens accepted')
    residues = hidden[valid]
    if not bool(torch.isfinite(residues).all().item()):
        raise ValueError('Nonfinite residue representations')
    pooled = residues.float().mean(dim=0).detach().cpu()
    if not bool(torch.isfinite(pooled).all().item()):
        raise ValueError('Nonfinite pooled representation')
    return pooled.tolist()


class ESM2Runtime:
    """Pinned local ESM-2 backbone; retains model/tokenizer only, never sequences or embeddings."""

    def __init__(self) -> None:
        """Load the pinned checkpoint after verifying manifest, hashes, and config.

        Purpose: guarantees inference runs only on the approved
            facebook/esm2_t12_35M_UR50D revision with intact local files
            (local_files_only, safetensors, no remote code). Prefers MPS
            when usable, otherwise CPU, recording any fallback.
        Raises:
            ValueError: on unapproved checkpoint/revision, assets outside
                models/, missing files, hash mismatch, or config mismatch.
        Flow: Receives pinned MODEL_PROVENANCE.json and local assets; returns None after preparing encode() for local inference."""
        import torch
        from transformers import EsmModel, EsmTokenizer
        manifest = json.loads((ROOT / 'MODEL_PROVENANCE.json').read_text())
        if (manifest['checkpoint'] != 'facebook/esm2_t12_35M_UR50D' or
                manifest['revision'] != '6fbf070e65b0b7291e7bbcd451118c216cff79d8'):
            raise ValueError('Unapproved checkpoint or revision')
        directory = ROOT / manifest['local_directory']
        if ROOT / 'models' not in directory.resolve().parents:
            raise ValueError('Model assets must remain inside the local model directory')
        required = {'config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt'}
        if set(manifest['files']) != required:
            raise ValueError('Checkpoint assets have not been explicitly prepared')
        for name, entry in manifest['files'].items():
            content = (directory / name).read_bytes()
            if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Local checkpoint integrity check failed')
        self.spec: ModelSpec = {key: manifest[key] for key in ModelSpec.__annotations__}
        if self.spec['dimension'] != 480 or self.spec['layer'] != 12 or self.spec['pooling'] != 'last_hidden_state_mean_residue_tokens' or self.spec['dtype'] != 'float32':
            raise ValueError('Unsupported ESM-2 representation specification')
        self.tokenizer = EsmTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = EsmModel.from_pretrained(str(directory), local_files_only=True,
                                             use_safetensors=True, add_pooling_layer=False)
        if self.model.config.hidden_size != 480 or self.model.config.num_hidden_layers != 12:
            raise ValueError('Backbone configuration mismatch')
        self.model.eval()
        self.model.requires_grad_(False)
        self.device, self.fallback = 'cpu', None
        if torch.backends.mps.is_available():
            try:
                torch.ones(1, device='mps').cpu()
                self.model.to('mps')
                self.device = 'mps'
            except (RuntimeError, NotImplementedError):
                self.model.to('cpu')
                self.fallback = 'MPS initialization failed; CPU used.'
        else:
            self.fallback = 'MPS unavailable; CPU used.'

    def encode(self, sequence: str) -> list[float]:
        """Encode one validated sequence to its mean-pooled vector.

        Purpose: tokenizes without truncation (token count must equal
            residues + 2) and runs inference mode forward. On MPS failure
            it moves to CPU once and recomputes, recording the fallback.
        Raises:
            ValueError: on tokenization/length mismatch.
            RuntimeError: when CPU inference itself fails.
        Flow: Receives validated local str from embed_protein(); returns residue-mean vector list[float] for embed_protein()."""
        import torch
        tokens = self.tokenizer(sequence, return_tensors='pt', return_special_tokens_mask=True,
                                truncation=False, padding=False)
        special = tokens.pop('special_tokens_mask')
        if tokens['input_ids'].shape[1] != len(sequence) + 2 or len(sequence) > MAX_RESIDUES:
            raise ValueError('Tokenization/length mismatch; refusing truncation')

        def forward() -> list[float]:
            inputs = {key: value.to(self.device) for key, value in tokens.items()}
            self.model.eval()
            with torch.inference_mode():
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
                return pool_residue_embeddings(hidden, inputs['attention_mask'], special.to(self.device), len(sequence))

        try:
            return forward()
        except (RuntimeError, NotImplementedError):
            if self.device != 'mps':
                raise
            self.model.to('cpu')
            self.device, self.fallback = 'cpu', 'MPS inference failed; recomputed on CPU.'
            return forward()


@lru_cache(maxsize=1)
def load_esm2_runtime() -> ESM2Runtime:
    """Return the cached process-wide ESM2Runtime, constructing it once.

    Flow: Receives external local-model configuration on first embed_protein() call; returns ESM2Runtime for embed_protein()."""
    return ESM2Runtime()


def _valid_embedding(embedding: ProteinEmbedding) -> bool:
    vector, spec = embedding['vector'], embedding['model']
    return (set(spec) == set(ModelSpec.__annotations__) and
            all(isinstance(spec[key], str) and bool(spec[key]) for key in ('checkpoint','revision','license','pooling','dtype')) and
            type(spec['layer']) is int and spec['layer'] > 0 and
            type(spec['dimension']) is int and spec['dimension'] > 0 and
            isinstance(vector, list) and len(vector) == spec['dimension'] and
            # A silent all-zero vector would make every cosine meaningless, so it is rejected here.
            all(finite_number(x) for x in vector) and any(x != 0 for x in vector))


def embed_protein(prepared: PreparedSequence, *, runtime: EmbeddingRuntime | None = None) -> ToolResult[ProteinEmbedding]:
    """Embed one validated local sequence with the pinned ESM-2 backbone.

    Purpose: produces a 480-dimension mean-pooled representation for
        similarity search. Re-verifies the sequence is unchanged since
        preparation (validation flag, length, SHA-256) so a swapped
        payload cannot be embedded under another identity.
    Args:
        prepared: PreparedSequence from prepare_protein_sequence().
        runtime: optional injected EmbeddingRuntime for tests; production
            uses the cached local runtime (excluded from JSON schema).
    Returns:
        ToolResult[ProteinEmbedding] with 'vector', 'model' spec,
            'device', 'fallback', identifiers, and provenance.
    Statuses:
        SUCCESS normally; ERROR when the input is invalid/changed or when
            load/inference fails (no substitute vector is generated).
    Limitations:
        Cosine similarity of these vectors is a COMPUTED representation
            metric, never measured activity or functional equivalence.
    Flow: Receives PreparedSequence from prepare_protein_sequence(); returns ToolResult[ProteinEmbedding] for compare_protein_embeddings() or find_similar_nrel_proteins()."""
    sources = prepared['provenance'].copy()
    sequence = prepared['normalized_sequence']
    fingerprint = sequence_fingerprint(sequence)
    if (prepared['validation'] != 'VALID' or not sequence or not set(sequence) <= CANONICAL
            or not 0 < len(sequence) <= MAX_RESIDUES or len(sequence) != prepared['sequence_length']
            or fingerprint['sha256'] != prepared['sequence_sha256']):
        return outcome('ERROR', None, sources, LIMITATIONS, 'COMPUTED', 'A validated, unchanged canonical sequence is required')
    try:
        engine = runtime if runtime is not None else load_esm2_runtime()
        vector = engine.encode(sequence)
        spec = engine.spec.copy()
        sources.append(model_source(spec))
        result: ProteinEmbedding = {key: prepared[key] for key in ('identifier', 'sequence_sha256', 'sequence_length', 'source_row', 'csv_sha256')}
        result.update(vector=vector, model=spec, device=engine.device, fallback=engine.fallback, provenance=sources)
        if not _valid_embedding(result):
            raise ValueError('Invalid embedding output')
        return outcome('SUCCESS', result, sources, LIMITATIONS, 'COMPUTED')
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return outcome('ERROR', None, sources, LIMITATIONS, 'COMPUTED',
                       'ESM-2 load/inference failed; prepare the pinned local assets and protein dependencies; no substitute embedding generated')


def compare_protein_embeddings(left: ProteinEmbedding, right: ProteinEmbedding) -> ToolResult[EmbeddingComparison]:
    """Compute cosine similarity between two same-model embeddings.

    Purpose: comparison primitive for similarity search. Requires
        identical model specs and non-degenerate vectors; scales inputs
        first to avoid overflow/underflow, then clamps to [-1, 1].
    Args:
        left: first ProteinEmbedding from embed_protein().
        right: second ProteinEmbedding from embed_protein().
    Returns:
        ToolResult[EmbeddingComparison] with 'cosine_similarity' and the
            shared 'model' spec.
    Statuses:
        SUCCESS normally; ERROR on incompatible/invalid embeddings.
    Limitations:
        No similarity thresholds exist; a high cosine implies nothing
            about activity, stability, or structure.
    Flow: Receives two ProteinEmbedding values from embed_protein(); returns ToolResult[EmbeddingComparison] for find_similar_nrel_proteins() or a final local comparison."""
    sources = left['provenance'].copy()
    sources.extend(source for source in right['provenance'] if source not in sources)
    try:
        if left['model'] != right['model'] or not _valid_embedding(left) or not _valid_embedding(right):
            raise ValueError('Incompatible or invalid embeddings')
        # Scale first to avoid finite-input overflow/underflow; no similarity thresholds.
        a, b = left['vector'], right['vector']
        scale_a, scale_b = max(abs(v) for v in a), max(abs(v) for v in b)
        a = [x / scale_a for x in a]
        b = [x / scale_b for x in b]
        cosine = math.fsum(x*y for x,y in zip(a,b)) / math.sqrt(math.fsum(x*x for x in a) * math.fsum(y*y for y in b))
        if not math.isfinite(cosine) or not -1.0000001 <= cosine <= 1.0000001:
            raise ValueError('Invalid cosine')
        result: EmbeddingComparison = {'cosine_similarity': max(-1.0, min(1.0, cosine)), 'model': left['model'].copy()}
        return outcome('SUCCESS', result, sources, LIMITATIONS, 'COMPUTED')
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
        return outcome('ERROR', None, sources, LIMITATIONS, 'COMPUTED', 'Incompatible dimensions/model/pooling or invalid numerical vectors')
