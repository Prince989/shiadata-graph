from src.extractors.chunkers import hadith_units
from src.extractors.epub_parser import ParsedUnit

def prepare(units: list[ParsedUnit]) -> list[ParsedUnit]:
    return hadith_units(units)
