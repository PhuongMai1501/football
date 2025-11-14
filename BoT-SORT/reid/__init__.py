# Local reid package init so modules can be imported via `import reid.*`.
# Expose TorchreIDInterface for convenience.
from .torchreid_interface import TorchreIDInterface

__all__ = ["TorchreIDInterface"]
