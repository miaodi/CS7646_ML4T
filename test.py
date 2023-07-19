
import numpy as np
import random as rand


Tc = np.ones((2, 3, 4))
print(np.sum(Tc, axis=2))
T = Tc/np.sum(Tc, axis=2, keepdims=True)

print(T)
T = np.cumsum(T, axis=2)
print(T)
print(np.argmax(T[1][2] > .56))

print(rand.randint(0, 5))
