"""Computer-vision pipeline for dog photo search.

Modules without torch imports (``spec``, ``weights``, ``pipeline``, ``text``) are safe to import
from the API process. Modules that import torch (``backbones``, ``gate``, ``heads``, ``bundle``,
``extractor``) are loaded only by the CV worker and the ML CLI.
"""
