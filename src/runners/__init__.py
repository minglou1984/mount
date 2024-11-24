REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .mount_parallel_runner import MOUNTParallelRunner
REGISTRY["mount_parallel"] = MOUNTParallelRunner
