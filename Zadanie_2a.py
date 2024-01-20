import numpy as np
import matplotlib.pyplot as plt


epsilon1, gamma1, h1 = 1.25, 0.5, 0.1
epsilon2, gamma2, h2 = 0.5, 0.2, 0.2


h = 0.01
t = np.arange(0, 50, h)

N1 = np.zeros(t.shape[0])
N2 = np.zeros(t.shape[0])

N1[0], N2[0] = 3, 4


for i in range(1, t.shape[0]):
    dN1_dt = (epsilon1 - gamma1 * (h1 * N1[i-1] + h2 * N2[i-1])) * N1[i-1]
    dN2_dt = (epsilon2 - gamma2 * (h1 * N1[i-1] + h2 * N2[i-1])) * N2[i-1]

    # Metoda Rungego-Kutty (dokładniejsza niż metoda Eulera)
    k1_N1 = h * dN1_dt
    k1_N2 = h * dN2_dt

    k2_N1 = h * ((epsilon1 - gamma1 * (h1 * (N1[i-1] + 0.5 * k1_N1) + h2 * (N2[i-1] + 0.5 * k1_N2))) * (N1[i-1] + 0.5 * k1_N1))
    k2_N2 = h * ((epsilon2 - gamma2 * (h1 * (N1[i-1] + 0.5 * k1_N1) + h2 * (N2[i-1] + 0.5 * k1_N2))) * (N2[i-1] + 0.5 * k1_N2))

    k3_N1 = h * ((epsilon1 - gamma1 * (h1 * (N1[i-1] + 0.5 * k2_N1) + h2 * (N2[i-1] + 0.5 * k2_N2))) * (N1[i-1] + 0.5 * k2_N1))
    k3_N2 = h * ((epsilon2 - gamma2 * (h1 * (N1[i-1] + 0.5 * k2_N1) + h2 * (N2[i-1] + 0.5 * k2_N2))) * (N2[i-1] + 0.5 * k2_N2))

    k4_N1 = h * ((epsilon1 - gamma1 * (h1 * (N1[i-1] + k3_N1) + h2 * (N2[i-1] + k3_N2))) * (N1[i-1] + k3_N1))
    k4_N2 = h * ((epsilon2 - gamma2 * (h1 * (N1[i-1] + k3_N1) + h2 * (N2[i-1] + k3_N2))) * (N2[i-1] + k3_N2))

    N1[i] = N1[i-1] + (k1_N1 + 2*k2_N1 + 2*k3_N1 + k4_N1) / 6
    N2[i] = N2[i-1] + (k1_N2 + 2*k2_N2 + 2*k3_N2 + k4_N2) / 6

    N1[i] = max(N1[i], 0)
    N2[i] = max(N2[i], 0)

    if N1[i] == 0 and N2[i] == 0:
        break


fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)

plt.plot(t, N1, color='b', label='N1')
plt.plot(t, N2, color='r', label='N2')
plt.xlabel('Czas')
plt.ylabel('Liczba osobników')
plt.legend()
plt.show()

# Wnioski: populacje po pewnym czsie osiągają stan równowagi, 
#          populacja N1 jest większa niż populacja N2

