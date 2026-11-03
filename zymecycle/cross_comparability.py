"""Strict cross-source comparison contracts; publication overlap never proves duplication."""
from __future__ import annotations

from .activity_types import AssayReference, CrossComparability, ExperimentDescriptor, ProteinVerification
from .contracts import Observation, ToolResult
from .results import outcome

REQUIRED = ('substrate', 'temperature_c', 'ph', 'duration_value', 'duration_unit', 'form', 'morphology',
            'crystallinity_percent', 'pretreatment', 'enzyme_amount_value', 'enzyme_amount_unit',
            'substrate_amount_value', 'substrate_amount_unit', 'reaction_volume_value', 'reaction_volume_unit',
            'enzyme_features', 'buffer', 'detection_methods', 'measurement_definition', 'analyte',
            'measurement_unit', 'replicate_n', 'uncertainty_value', 'uncertainty_type')
LIMITATIONS = ['Comparability is for a direct same-condition response comparison with documented equivalence, not a causal or predictive claim.',
               'Strict field matching performs no unit conversion, protocol synonym reconciliation or cross-form pooling.',
               'Missing documentation is not evidence that a procedure was absent; parent sequence identity does not establish assay-variant equivalence.',
               'Shared publication or nominal conditions do not prove duplicate experiments; no source measurements are combined.']


def describe_nrel_observation(observation: Observation) -> ExperimentDescriptor:
    """Describe an NREL observation in the shared cross-source field vocabulary.

    Purpose: maps temperature/pH/form (plus PET substrate and amorphous
        morphology for film) into an ExperimentDescriptor; every other
        required field stays absent (unknown), never defaulted.
    Flow: Receives Observation from parse_nrel(); returns ExperimentDescriptor for assess_cross_source_comparability(), retaining undocumented fields as unknown."""
    return {'protein_id': observation['enzyme_identifier'], 'publication_doi': None, 'experiment_locator': None,
            'fields': {'substrate': 'PET', 'temperature_c': observation['temperature_c'], 'ph': observation['ph'],
                       'form': 'powder' if observation['pet_form'] == 'CRYSTALLINE_POWDER' else 'film',
                       'morphology': 'amorphous' if observation['pet_form'] == 'AMORPHOUS_FILM' else None},
            'source': {**observation['source'], 'locator': f"{observation['source']['locator']}#record={observation['source_row']}&column={observation['source_column']}"}}


def describe_pazy_assay(reference: AssayReference, measurement_index: int = 0) -> ExperimentDescriptor:
    """Describe one PAZy assay measurement in the shared field vocabulary.

    Purpose: projects the assay's conditions/material plus the selected
        measurement's definition, analyte, unit, replicates, and
        uncertainty into an ExperimentDescriptor.
    Raises:
        ValueError: when the measurement index is outside the assay.
    Flow: Receives AssayReference from iter_assays() and selected measurement_index int; returns ExperimentDescriptor for assess_cross_source_comparability()."""
    assay = reference['assay']
    measurements = assay['measurements']
    if type(measurement_index) is not int or measurement_index < 0 or measurement_index >= max(1, len(measurements)):
        raise ValueError('Measurement selection outside assay')
    measurement = measurements[measurement_index] if measurements else {}
    fields = {k: v for k, v in assay['conditions'].items() if k in REQUIRED}
    fields.update({k: v for k, v in assay['material'].items() if k in REQUIRED})
    fields.update(substrate=assay['substrate'], detection_methods=assay['detection_methods'],
                  measurement_definition=measurement.get('metric_type'), analyte=measurement.get('analyte'),
                  measurement_unit=measurement.get('unit'), replicate_n=measurement.get('replicate_n'),
                  uncertainty_value=measurement.get('uncertainty_value'), uncertainty_type=measurement.get('uncertainty_type'),
                  qualifier=measurement.get('qualifier'), numeric_response_available=type(measurement.get('value')) in (int, float))
    return {'protein_id': str(reference['protein_id']), 'publication_doi': reference['publication_doi'],
            'experiment_locator': assay['evidence'].get('location'), 'fields': fields, 'source': reference['source']}


def assess_cross_source_comparability(nrel: ExperimentDescriptor, pazy: ExperimentDescriptor,
                                      verification: ProteinVerification | None = None) -> ToolResult[CrossComparability]:
    """Strictly assess whether an NREL/PAZy pair supports direct comparison.

    Purpose: exact-matches 24 required fields with no unit conversion,
        synonym reconciliation, or pooling. Any incompatibility means
        NOT_COMPARABLE; missing fields or unverified identity cap the
        result at LIMITED_COMPARISON or INSUFFICIENT_INFORMATION; only
        a fully documented exact match with proven identity reaches
        VERIFIED_COMPARABLE. Same-publication pairs are flagged as
        potential duplicates, never replication.
    Args:
        nrel: ExperimentDescriptor from describe_nrel_observation().
        pazy: ExperimentDescriptor from describe_pazy_assay().
        verification: optional bound ProteinVerification for this pair.
    Returns:
        ToolResult[CrossComparability] with category, missing/
            incompatible fields, identity status, and duplication flags.
    Flow: Receives ExperimentDescriptor values from describe_nrel_observation()/describe_pazy_assay() and optional verify_snapshot_identities() ProteinVerification; returns ToolResult[CrossComparability] as the final strict audit assessment."""
    left, right = nrel['fields'], pazy['fields']
    missing, incompatible = [], []
    for field in REQUIRED:
        a, b = left.get(field), right.get(field)
        if a is None or b is None or a in ('', [], {}) or b in ('', [], {}):
            missing.append(field)
        elif a != b:
            incompatible.append(field)
    identity = 'UNVERIFIED'
    if (verification is not None and verification['nrel_identifier'] == nrel['protein_id']
            and str(verification['pazy_protein_id']) == pazy['protein_id'] and verification['provenance']):
        identity = verification['identity']
    if identity == 'DIFFERENT_SEQUENCE': incompatible.append('protein_sequence')
    if identity != 'EXACT_SEQUENCE_MATCH': missing.append('verified_protein_identity')
    if right.get('numeric_response_available') is False: missing.append('numeric_pazy_response')
    if right.get('qualifier') not in (None, '', '='): missing.append('uncensored_pazy_response')
    if incompatible:
        category = 'NOT_COMPARABLE'
    elif identity != 'EXACT_SEQUENCE_MATCH' or any(f in missing for f in ('substrate','temperature_c','ph','numeric_pazy_response')):
        category = 'INSUFFICIENT_INFORMATION'
    elif missing:
        category = 'LIMITED_COMPARISON'
    else:
        category = 'VERIFIED_COMPARABLE'
    pub_a, pub_b = nrel['publication_doi'], pazy['publication_doi']
    same_pub = bool(pub_a and pub_b and pub_a.casefold() == pub_b.casefold())
    same_experiment = (same_pub and identity == 'EXACT_SEQUENCE_MATCH' and not incompatible and not missing
                       and bool(nrel['experiment_locator']) and nrel['experiment_locator'] == pazy['experiment_locator'])
    duplicate = ('SAME_DOCUMENTED_EXPERIMENT' if same_experiment else 'SHARED_PUBLICATION_ONLY' if same_pub else
                 'DISTINCT_PUBLICATIONS' if pub_a and pub_b else 'UNRESOLVED')
    result: CrossComparability = {'category': category, 'missing_fields': missing, 'incompatible_fields': incompatible,
        'identity_status': identity, 'duplicate_status': duplicate,
        'quantitative_comparison_supported': category == 'VERIFIED_COMPARABLE', 'causal_claim_supported': False}
    sources = [nrel['source'], pazy['source']]
    for source in verification['provenance'] if verification else []:
        if source not in sources: sources.append(source)
    return outcome('SUCCESS', result, sources, LIMITATIONS, 'COMPUTED')
