"""OpenAPI / Swagger specification parser package."""

from api_analyzer.parser.classifier import classify
from api_analyzer.parser.ingestor import SpecParseError, ingest, parse_spec_dict

__all__ = ["SpecParseError", "classify", "ingest", "parse_spec_dict"]
