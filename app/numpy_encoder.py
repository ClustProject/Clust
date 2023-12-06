import json
import numpy as np

# for json error
# TypeError: Object of type [float32 | int64] is not JSON serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)