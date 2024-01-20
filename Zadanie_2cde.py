import numpy as np
import matplotlib.pyplot as plt


def population_model(epsilon1, gamma1, h1, epsilon2, gamma2, h2, initial_conditions):
    h = 0.001
    t = np.arange(0, 50, h)

    N1 = np.zeros(t.shape[0])
    N2 = np.zeros(t.shape[0])

    N1[0], N2[0] = initial_conditions

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

    return N1, N2


epsilon1, gamma1, h1 = 0.8, 1, 0.3
epsilon2, gamma2, h2 = 0.4, 0.5, 0.4

initial_conditions_c = (4, 8)
initial_conditions_d = (8, 8)
initial_conditions_e = (12, 8)

N1_c, N2_c = population_model(epsilon1, gamma1, h1, epsilon2, gamma2, h2, initial_conditions_c)
N1_d, N2_d = population_model(epsilon1, gamma1, h1, epsilon2, gamma2, h2, initial_conditions_d)
N1_e, N2_e = population_model(epsilon1, gamma1, h1, epsilon2, gamma2, h2, initial_conditions_e)

fig, ax = plt.subplots()
ax.plot(N1_c, N2_c, label='(N1, N2) = (4, 8)')
ax.plot(N1_d, N2_d, label='(N1, N2) = (8, 8)')
ax.plot(N1_e, N2_e, label='(N1, N2) = (12, 8)')
ax.scatter([4, 8, 12], [8, 8, 8], color='r', label='Warunki początkowe', marker='o')
ax.set_xlabel('Liczba osobników populacji N1')
ax.set_ylabel('Liczba osobników populacji N2')
ax.set_title('Portret fazowy dla różnych warunków początkowych')
ax.legend()
ax.grid(True)  
plt.show()

# Dla wszystkich warunków początkowych trajektorie populacji N1 i N2 zbiegają do punktu równowagi, 
#gdzie liczba osobników w obu populacjach jest utrzymywana na stałym poziomie.
# Dla warunków początkowych (8,8) populacje osiągają równowagę, N1 nie dominuje nad N2 i odwrotnie
# Dla warunków początkowych (4,8) i (12,8) populacja również osiągają stan równowagi,
#jednakże populacja N2 jest większa niż N1 w stanie końcowym




