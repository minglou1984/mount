REGISTRY = {}

from .basic_controller import BasicMAC
from .decamd_controller import DecAMDMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["decamd_mac"] = DecAMDMAC
