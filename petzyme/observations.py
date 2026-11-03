"""NREL recorded observations, exact-condition selection and descriptive summaries."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .contracts import (
    ConditionSummary,
    NrelProtein,
    NrelTable,
    Observation,
    ParsedObservations,
    PetForm,
    Source,
    ToolResult,
)
from .identity import sequence_fingerprint
from .nrel import DOI, RECORD_URL, NrelError, parse_csv_text
from .results import finite_number, outcome, sources_from_observations

LIMITATIONS = [
    'Original CSV unit declarations are UNKNOWN/UNVERIFIED. Original values remain unchanged; supported working-unit representations are separately labelled INFERRED and COMPUTED.',
    'Blank cells are missing observations; reported zeros do not prove inactivity.',
    'Summaries describe recorded responses within one exact condition, not predicted performance.',
    'Crystalline powder and amorphous film must be assessed separately.',
    'Numerical floats are conveniences; original_text preserves exact source precision.',
    'Sequence hashes require unambiguous ungapped residues; sequence_reference always locates the original CSV cell.',
]
FORMS: dict[str, PetForm] = {'cryPow': 'CRYSTALLINE_POWDER', 'aFilm': 'AMORPHOUS_FILM'}
CONDITION_COLUMN = re.compile(r'activity_at_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(cryPow|aFilm)')


def load_nrel(csv_path: str = 'data/nrel/label_data.csv') -> ToolResult[NrelTable]:
    """Load the local NREL CSV as original strings with a SHA-256 fingerprint.

    Purpose: entry point for all local experimental evidence. Reads the CSV
        exactly as stored (no value conversion) and hashes the raw bytes so
        every downstream observation can prove which file version it came from.
    Args:
        csv_path: filesystem path to label_data.csv. Defaults to the pinned
            local copy at 'data/nrel/label_data.csv'.
    Returns:
        ToolResult[NrelTable]; result holds 'columns', 'rows' (all strings),
            'source' (NREL DOI/attribution), and 'csv_sha256'.
    Statuses:
        SUCCESS when rows exist; NO_DATA when the CSV has headers only;
        ERROR when the file is missing, undecodable, or malformed.
    Example:
        >>> out = load_nrel('data/nrel/label_data.csv')
        >>> out['status'], out['result']['columns'][:2]
        ('SUCCESS', ['', 'sequence'])
    Limitations:
        Blank vs zero handling, units, and condition parsing happen later;
        this step only preserves original cells.
    Flow: Receives external NREL CSV path str; returns ToolResult[NrelTable] for parse_nrel() or select_nrel_protein()."""
    source: Source = {
        'name': 'NREL', 'url': RECORD_URL, 'doi': DOI, 'locator': csv_path,
        'attribution': 'Norton-Baker et al., Activity across temperature and pH of PET hydrolase candidates. CC BY 4.0.',
        'retrieved_at_utc': None,
    }
    try:
        content = Path(csv_path).read_bytes()
        columns, rows = parse_csv_text(content.decode('utf-8'))
        digest = hashlib.sha256(content).hexdigest()
        root = Path(__file__).resolve().parents[1]
        if Path(csv_path).resolve() == root/'data/nrel/label_data.csv':
            manifest = json.loads((root/'NREL_SOURCE_MANIFEST.json').read_text())
            if digest != manifest['unit_reconciliation']['csv_sha256']:
                raise NrelError('Pinned original CSV fingerprint mismatch')
    except (OSError, UnicodeError, csv.Error, NrelError, ValueError, KeyError, TypeError):
        return outcome('ERROR', None, [source], LIMITATIONS, 'MEASURED', 'Unable to load a valid local NREL CSV')
    table: NrelTable = {'columns': columns, 'rows': rows, 'source': source, 'csv_sha256': digest}
    return outcome('SUCCESS' if rows else 'NO_DATA', table, [source], LIMITATIONS, 'MEASURED')


def _table_error(table: NrelTable) -> str | None:
    """Validate the minimum actual NREL schema without changing cell values."""
    columns = table['columns']
    if not {'', 'sequence'} <= set(columns) or len(set(columns)) != len(columns):
        return 'Expected unique original headers including the unnamed identifier and sequence columns'
    if any(len(row) != len(columns) or any(not isinstance(cell, str) for cell in row)
           for row in table['rows']):
        return 'Expected original string cells with consistent row widths'
    return None


def parse_nrel(table: NrelTable) -> ToolResult[ParsedObservations]:
    """Parse original CSV cells into typed observations with exact text preserved.

    Purpose: turns each non-blank activity_at_<pH>_<temp>_<form> cell into
        one Observation carrying its enzyme, condition, original string,
        float convenience value, sequence hash, and cell locator. Blank
        cells become missing_cells counts, never zeros.
    Args:
        table: NrelTable from load_nrel() with unique headers including
            the unnamed identifier ('') and 'sequence' columns.
    Returns:
        ToolResult[ParsedObservations]; result holds 'observations' and
            'missing_cells'. Each observation keeps 'original_text' verbatim
            plus 'numerical_value', 'is_zero', 'unit_status'
            ('UNKNOWN/UNVERIFIED'), and 'evidence_category' ('MEASURED').
    Statuses:
        SUCCESS normally; NO_DATA when no condition cells are filled;
        ERROR on schema violations or non-numeric cell text.
    Example:
        >>> parsed = parse_nrel(load_nrel()['result'])
        >>> obs = parsed['result']['observations'][0]
        >>> (obs['source_row'], obs['source_column'], obs['unit_status'])
        (2, 'activity_at_5.5_40_cryPow', 'UNKNOWN/UNVERIFIED')
    Limitations:
        Floats are conveniences; Decimal(original_text) is authoritative.
        Zeros are recorded responses, not proof of inactivity.
    Flow: Receives NrelTable from load_nrel(); returns ToolResult[ParsedObservations] for filter_observations() or summarize_condition()."""
    error = _table_error(table)
    conditions: list[tuple[int, str, float, float, PetForm]] = []
    for index, column in enumerate(table['columns']):
        if not column.startswith('activity_at_'):
            continue
        match = CONDITION_COLUMN.fullmatch(column)
        if match is None:
            error = 'Unrecognized experimental-condition column; no values interpreted'
            break
        ph, temperature, form = match.groups()
        if not math.isfinite(float(ph)) or not math.isfinite(float(temperature)):
            error = 'Experimental-condition value is not finite'
            break
        conditions.append((index, column, float(ph), float(temperature), FORMS[form]))
    if error or not conditions:
        return outcome('ERROR', None, [table['source']], LIMITATIONS, 'MEASURED',
                       error or 'No experimental-condition columns found')
    observations: list[Observation] = []
    missing = 0
    id_index, sequence_index = table['columns'].index(''), table['columns'].index('sequence')
    for row_number, row in enumerate(table['rows'], start=2):
        fingerprint = sequence_fingerprint(row[sequence_index])
        for index, column, ph, temperature, form in conditions:
            text = row[index]
            if not text.strip():
                missing += 1
                continue
            try:
                exact = Decimal(text.strip())
                value = float(exact)
                # A nonzero decimal that underflows to 0.0 would fabricate a false zero response.
                if not math.isfinite(value) or (value == 0 and exact != 0):
                    raise ValueError('Non-finite or unrepresentable value')
            except (InvalidOperation, ValueError, OverflowError):
                return outcome('ERROR', None, [table['source']], LIMITATIONS, 'MEASURED',
                               f'Invalid recorded numerical value at record {row_number}, column {column}')
            observation: Observation = {
                'enzyme_identifier': row[id_index] or None, 'sequence_sha256': fingerprint['sha256'],
                'sequence_reference': f'{table["source"]["locator"]}#record={row_number}&column=sequence',
                'source_row': row_number, 'source_column': column, 'source': table['source'],
                'csv_sha256': table['csv_sha256'], 'temperature_c': temperature, 'ph': ph,
                'pet_form': form, 'original_text': text, 'numerical_value': value,
                'unit_status': 'UNKNOWN/UNVERIFIED', 'is_zero': exact == 0, 'evidence_category': 'MEASURED',
            }
            observations.append(observation)
    result: ParsedObservations = {'observations': observations, 'missing_cells': missing}
    return outcome('SUCCESS' if observations else 'NO_DATA', result, [table['source']],
                   LIMITATIONS, 'MEASURED')


def filter_observations(observations: list[Observation], *, enzyme_identifier: str | None = None,
                        temperature_c: float | None = None, ph: float | None = None,
                        pet_form: PetForm | None = None) -> ToolResult[list[Observation]]:
    """Select observations by exact enzyme/condition match; no ranges or coercion.

    Purpose: narrows parsed observations to one exact experimental slice.
        Equality is strict (40 == 40.0 in Python, but '40' or NaN are
        rejected) so near-miss conditions can never silently merge.
    Args:
        observations: list from parse_nrel().
        enzyme_identifier: exact label such as 'ESM050', or None for all.
        temperature_c: exact degrees Celsius, or None.
        ph: exact pH, or None.
        pet_form: 'CRYSTALLINE_POWDER' or 'AMORPHOUS_FILM', or None.
    Returns:
        ToolResult[list[Observation]] with the matching observations.
    Statuses:
        SUCCESS when at least one matches; NO_DATA when none do;
        ERROR on invalid filter types or unknown PET form.
    Example:
        >>> obs = parse_nrel(load_nrel()['result'])['result']['observations']
        >>> out = filter_observations(obs, enzyme_identifier='ESM050',
        ...     temperature_c=60, ph=6.5, pet_form='CRYSTALLINE_POWDER')
        >>> out['status'], out['result'][0]['original_text']
        ('SUCCESS', '0.13')
    Flow: Receives list[Observation] from parse_nrel() and external filter values; returns ToolResult[list[Observation]] for summarize_condition() or final selection."""
    sources = sources_from_observations(observations)
    if (pet_form is not None and pet_form not in FORMS.values()) or any(
        value is not None and not finite_number(value)
        for value in (temperature_c, ph)
    ) or (enzyme_identifier is not None and not isinstance(enzyme_identifier, str)):
        return outcome('ERROR', None, sources, LIMITATIONS, 'MEASURED', 'Invalid exact-condition filter')
    matches = [observation for observation in observations if
               (enzyme_identifier is None or observation['enzyme_identifier'] == enzyme_identifier) and
               (temperature_c is None or observation['temperature_c'] == temperature_c) and
               (ph is None or observation['ph'] == ph) and
               (pet_form is None or observation['pet_form'] == pet_form)]
    return outcome('SUCCESS' if matches else 'NO_DATA', matches, sources, LIMITATIONS, 'MEASURED')


def summarize_condition(observations: list[Observation], temperature_c: float,
                        ph: float, pet_form: PetForm) -> ToolResult[ConditionSummary]:
    """Describe recorded responses within one exact condition.

    Purpose: answers "what was recorded here?" with counts and the top-5
        responses ranked by exact source decimals (Decimal, not float), so
        display rounding can never reorder the ranking.
    Args:
        observations: list from parse_nrel().
        temperature_c: exact degrees Celsius.
        ph: exact pH.
        pet_form: 'CRYSTALLINE_POWDER' or 'AMORPHOUS_FILM'.
    Returns:
        ToolResult[ConditionSummary]; result holds 'condition',
            'observation_count', 'recorded_zeros',
            'positive_recorded_responses', and 'leading_responses'
            (up to 5 observations, ties by identifier/row/column).
    Statuses:
        SUCCESS when the condition has observations; NO_DATA otherwise;
        ERROR on invalid condition values.
    Example:
        >>> obs = parse_nrel(load_nrel()['result'])['result']['observations']
        >>> out = summarize_condition(obs, 60, 6.5, 'CRYSTALLINE_POWDER')
        >>> out['result']['observation_count'] > 0
        True
    Limitations:
        A larger recorded response is not demonstrated activity
        superiority; units are unverified and no significance is computed.
    Flow: Receives list[Observation] from parse_nrel() and external exact condition values; returns ToolResult[ConditionSummary] as a final descriptive summary."""
    filtered = filter_observations(observations, temperature_c=temperature_c, ph=ph, pet_form=pet_form)
    if filtered['status'] == 'ERROR':
        return outcome('ERROR', None, filtered['provenance'], LIMITATIONS, 'COMPUTED', filtered['message'])
    matches = filtered['result'] or []
    # Negated Decimal ranks exact values descending while tiebreaks stay ascending; floats never order this.
    leading = sorted(matches, key=lambda observation: (
        Decimal(observation['original_text'].strip()).copy_negate(), observation['enzyme_identifier'] or '',
        observation['source_row'], observation['source_column']))[:5]
    result: ConditionSummary = {
        'condition': {'temperature_c': temperature_c, 'ph': ph, 'pet_form': pet_form},
        'observation_count': len(matches), 'recorded_zeros': sum(o['is_zero'] for o in matches),
        'positive_recorded_responses': sum(Decimal(o['original_text'].strip()) > 0 for o in matches),
        'leading_responses': leading,
    }
    return outcome('SUCCESS' if matches else 'NO_DATA', result, filtered['provenance'],
                   LIMITATIONS, 'COMPUTED', None if matches else 'No observations at this exact condition')


def select_nrel_protein(table: NrelTable, enzyme_identifier: str) -> ToolResult[NrelProtein]:
    """Fetch one NREL protein row by exact identifier.

    Purpose: resolves a label such as 'ESM050' to its source row and
        stored sequence for identity comparison. Ambiguous labels fail
        instead of guessing, because a wrong row would corrupt any
        downstream sequence check.
    Args:
        table: NrelTable from load_nrel().
        enzyme_identifier: nonempty exact label from the '' column.
    Returns:
        ToolResult[NrelProtein]; result holds 'enzyme_identifier',
            'amino_acid_sequence' (None when the cell is blank),
            'source_row', 'source', and 'csv_sha256'.
    Statuses:
        SUCCESS on exactly one match; NO_DATA when absent; ERROR when the
        identifier is empty or matches multiple rows.
    Example:
        >>> out = select_nrel_protein(load_nrel()['result'], 'ESM050')
        >>> out['status'], out['result']['source_row']
        ('SUCCESS', 160)
    Flow: Receives NrelTable from load_nrel() and external enzyme identifier str; returns ToolResult[NrelProtein] for compare_identity()."""
    error = _table_error(table)
    if not isinstance(enzyme_identifier, str) or not enzyme_identifier:
        error = 'Expected a nonempty exact enzyme identifier'
    if error:
        return outcome('ERROR', None, [table['source']], LIMITATIONS, 'MEASURED', error)
    id_index = table['columns'].index('')
    matches = [(index, row) for index, row in enumerate(table['rows'], start=2)
               if row[id_index] == enzyme_identifier]
    if not matches:
        return outcome('NO_DATA', None, [table['source']], LIMITATIONS, 'MEASURED', 'No exact enzyme identifier match')
    if len(matches) > 1:
        return outcome('ERROR', None, [table['source']], LIMITATIONS, 'MEASURED', 'Ambiguous enzyme identifier')
    number, row = matches[0]
    protein: NrelProtein = {
        'enzyme_identifier': enzyme_identifier, 'amino_acid_sequence': row[table['columns'].index('sequence')] or None,
        'source_row': number, 'source': table['source'], 'csv_sha256': table['csv_sha256'],
    }
    return outcome('SUCCESS', protein, [table['source']], LIMITATIONS, 'MEASURED')
