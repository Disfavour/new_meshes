import os
print(os.cpu_count())
print(os.environ['OPENBLAS_NUM_THREADS'])

import numpy
numpy.show_config()