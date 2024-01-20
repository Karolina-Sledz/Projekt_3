import numpy as np
import matplotlib.pyplot as plt


K = 100000
t0, x0, r = 75, 10, 0.4
l = r

dt = 0.1
t_max = 150


t_values = [t0]
x_gompertz = [x0]
x_verhulst = [x0]


for t in np.arange(t0, t_max, dt):
    # Model Gompertza
    dx_gompertz = r * x_gompertz[-1] * np.log(K / x_gompertz[-1])
    x_gompertz.append(x_gompertz[-1] + dx_gompertz * dt)
    
 
    # Model Verhulsta
    dx_verhulst = l * x_verhulst[-1] * (1 - x_verhulst[-1] / K)
    x_verhulst.append(x_verhulst[-1] + dx_verhulst * dt)
    
    t_values.append(t + dt)
    


fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)

plt.plot(t_values, x_gompertz, label='Gompertz Model')
plt.plot(t_values, x_verhulst, label='Verhulst Model')

plt.xlabel('Czas')
plt.ylabel('Rozmiar guza')
plt.legend()
plt.show()




