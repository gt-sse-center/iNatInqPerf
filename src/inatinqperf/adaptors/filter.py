from collections.abc import Sequence


class Filter:
    """A class encapsulating the filter for a query. Although this is extremely simple, it can be extended to include more complex filters in the future."""

    def __init__(self, acceptable_iconic_groups: Sequence[str]) -> None:
        """Constructor for the filter."""
        self.acceptable_iconic_groups = acceptable_iconic_groups
