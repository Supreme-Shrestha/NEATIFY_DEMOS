
import neatify
import inspect
from neatify.distributed.master import DistributedPopulation

try:
    print(inspect.getsource(DistributedPopulation.__init__))
except Exception as e:
    print(f"Error getting source: {e}")
