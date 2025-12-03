

class Filter:
    """A class encapsulating the filter for a query. Although this is extremely simple, it can be extended to include more complex filters in the future."""

    def __init__(self, min_id: int, max_id: int) -> None:
        """Constructor for the filter."""
        self.min_id = min_id
        self.max_id = max_id
