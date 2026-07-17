__all__ = ["Indexer", "Searcher", "Checkpoint"]


def __getattr__(name):
    if name == "Indexer":
        from .indexer import Indexer
        return Indexer
    if name == "Searcher":
        from .searcher import Searcher
        return Searcher
    if name == "Checkpoint":
        from .modeling.checkpoint import Checkpoint
        return Checkpoint
    raise AttributeError(name)
