class PercolationFailed(Exception):
    """
    Raised when the electronic percolation fraction is below min_threshold.
    Caller should catch this, increment seed, and regenerate from Step 2.

    Attributes:
      run_id  : run identifier from sim.run_id
      seed    : the seed that produced this failure
      fraction: the percolation fraction that was measured
      threshold: the required minimum
    """

    def __init__(
        self,
        run_id: int,
        seed: int,
        fraction: float,
        threshold: float,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.fraction = fraction
        self.threshold = threshold
        super().__init__(
            f"Run {run_id} seed={seed}: electronic percolation fraction "
            f"{fraction:.4f} < threshold {threshold:.4f}. Retry with new seed."
        )
