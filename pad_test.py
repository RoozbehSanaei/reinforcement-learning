import numpy as np
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
a = np.arange(9)
a = a.reshape((3, 3))
np.pad(a, 3, pad_with,padder=0)
