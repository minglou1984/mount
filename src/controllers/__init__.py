REGISTRY = {}

from .basic_controller import BasicMAC
from .mount_controller import MOUNTMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mount_mac"] = MOUNTMAC