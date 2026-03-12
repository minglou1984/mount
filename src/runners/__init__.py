REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .decamd_parallel_runner import DecAMDParallelRunner
REGISTRY["decamd_parallel"] = DecAMDParallelRunner
