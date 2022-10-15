from typing import Iterable, Sequence

import luigi


class CompositeTarget(luigi.Target):
    """Combination of multiple targets."""

    def __init__(self, inputs: Iterable[luigi.Target]):
        self._inputs: Sequence[luigi.Target] = tuple(inputs)

    def exists(self) -> bool:
        """Check subtasks"""
        return all(input.exists() for input in self._inputs)


class ConstTarget(luigi.Target):
    """Target with constant state."""

    def __init__(self, exists: bool):
        self._exists: bool = exists

    def exists(self) -> bool:
        return self._exists
