"""Live smoke test. Emits only status and schema; never record values or IDs."""
from datetime import datetime, timezone
from petzyme.pazy import PazyClient, PazyError, SOURCE


def schema(record: dict) -> dict[str, str]:
    """Flow: Receives transient record dict from main(); returns dict[str, str] for schema-only printing by main()."""
    return {key: type(value).__name__ for key, value in record.items()}


def main() -> None:
    """Flow: Receives external CLI invocation; returns None after printing only PAZy status and field types."""
    print("Source:", SOURCE)
    print("Checked at UTC:", datetime.now(timezone.utc).isoformat())
    client = PazyClient()
    try:
        proteins = client.search_pet_proteins(limit=1)
        print("PET protein search: OK; returned", len(proteins), "record(s)")
        if proteins:
            print("Protein list field types:", schema(proteins[0]))
            protein = client.get_protein(proteins[0]["id"])
            print("Protein detail: OK; field types:", schema(protein))
            del protein
        else:
            print("Protein detail: not checked (no search results)")
        del proteins
    except PazyError as error:
        print("Protein smoke test failed:", str(error))
    try:
        entries = client.get_literature(limit=1)
        print("Literature: OK; returned", len(entries), "record(s)")
        if entries:
            print("Literature field types:", schema(entries[0]))
        del entries
    except PazyError as error:
        print("Literature smoke test failed:", str(error))


if __name__ == "__main__":
    main()
