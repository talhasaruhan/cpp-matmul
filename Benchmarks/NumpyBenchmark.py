import numpy as np
import time

n = 1000
a = np.random.randn(n, n)*50

start = time.time()
b = np.dot(a, a)
end = time.time()

print(end-start)