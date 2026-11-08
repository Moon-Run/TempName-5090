"""SM120 Flux/CUTLASS, TACO and arrival-priority CUDA execution."""
__version__ = '0.1.0'


def __getattr__(name):
    if name == 'Flux5090':
        from .runtime import Flux5090
        return Flux5090
    raise AttributeError(name)
