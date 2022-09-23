import re
import numpy as np


def KITTICameraMatrixFromString(str, camera_id):
    lines = re.search(f"(?<=P{camera_id}: ).*?(?=\n)", str).group(0).split(" ")
    lines = np.array(lines, dtype=np.float32)
    lines = np.reshape(lines, (3, 4))
    return lines

# Data Helpers
def KITTICameraMatrixFromTXT(filepath, camera_id):
    with open(filepath, 'r') as f:
        content = f.read()
        return KITTICameraMatrixFromString(content, camera_id)
