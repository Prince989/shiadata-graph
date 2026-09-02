from src.extractors.chunkers import tafsir_ayah_units
from src.extractors.epub_parser import ParsedUnit

def prepare(units: list[ParsedUnit]) -> list[ParsedUnit]:
    return tafsir_ayah_units(units)
