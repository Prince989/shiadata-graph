from src.extractors.chunkers import pack_history
from src.extractors.epub_parser import ParsedUnit

def prepare(
    units: list[ParsedUnit],
    pages_per_call: int,
    max_chars: int,
) -> list[ParsedUnit]:
    return pack_history(units, pages_per_call, max_chars)
