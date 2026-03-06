from dataclasses import dataclass, field
from typing import List

@dataclass
class DatasetEntry:
    """Data class for a dataset entry."""
    id_primeiro_acesso_historico: int
    ds_observacao: str

@dataclass
class Embeddings:
    """Data class for an embedding."""
    ds_observacao: List[str] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)

@dataclass
class EmbeddingsWithLabels:
    """Data class for an embedding with labels."""
    ds_observacao: List[str]
    embeddings: List[float]
    labels: List[int]