import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import math
sns.set()

# initial conditions (BE SURE TO ADJUST SOLUTIONS IF INITIAL CONDITIONS ARE CHANGED!)
y = 4
t = 0


h = .3 # step-size 
n = 10 # number of iterations


# Assortment of various ODEs and their solutions 
def fn_1(t, y):
    return (2*y-18*t)/(1+t)

def sol_1(t):
    return 4+8*t-5*t**2

def fn_2(t, y):
    return np.exp(-y)*(2*t-4)

def sol_2(t):
    return np.log(t**2-4*t+np.exp(4))

def fn_3(t, y):
    return 3*t*y

def sol_3(t):
    return 4*np.exp(((3/2)*t**2))


# functions and solutions encapsulated in lists for iteration purposes 
functions = [fn_1, fn_2, fn_3]
sols = [sol_1, sol_2, sol_3]

# empty lists for storing data
eu_y_vals = [[], [], []]
t_vals = [[], [], []]
rk4_y_vals = [[], [], []]
eu_error_vals = [[], [], []]
rk4_error_vals = [[], [], []]

# Recursively defined Euler Method Approximation of ODE 
# Appends euler y-values and t-values to listgs above 
def euler(fn, i, t, y):
    while len(eu_y_vals[i]) < n:
        eu_y_vals[i].append(y)
        t_vals[i].append(t)
        t += h
        y = y + fn(t, y)*h
        euler(fn, i, t, y)

# Recursively defined Runge-Kutta-4 Approximation of ODE
# appends rk4 y-values to lists above 
def rk4(fn, i, t, y):
    k1 = h * fn(t, y)
    k2 = h * fn(t+h/2, y+k1/2)
    k3 = h * fn(t+h/2, y+k2/2)
    k4 = h * fn(t+h, y+k3)
    while len(rk4_y_vals[i]) < n:
        rk4_y_vals[i].append(y)
        t += h
        y = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        rk4(fn, i, t, y)

# Generates slope field for given ODE       
def slope_field(fn):
    # controls size of slope field windows 
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 4, 10)
    # use x,y
    for j in x:
        for k in y:
            slope = fn(j, k)
            domain = np.linspace(j-0.1,j+0.1,10)
            def fun(x1,y1):
                z = slope*(domain-x1)+y1
                return z
            plt.plot(domain, fun(j,k), color='black')
    plt.grid(True)


# iterates through each function and calls approximation methods
# after each iteration, calculates local truncated error for
# both Euler and RK4 and appends error values to lists above 
for i in range(len(functions)):
    euler(functions[i], i, t, y)
    rk4(functions[i], i, t, y)
    for k in range(n):
        eu_error_vals[i].append(np.abs(eu_y_vals[i][k] - [sols[i](t) for t in t_vals[i]][k]))
        rk4_error_vals[i].append(np.abs(rk4_y_vals[i][k] - [sols[i](t) for t in t_vals[i]][k]))


# plots a 3 x 3 figure. Graphs in first column compare Euler Method Approximation
# and RK4 approximation to exact solution. Graphs in second column compare 
# truncated error for Euler method and Rk4. Third column includes vector fields
# for each ODE. 

fig = plt.figure()

# Column 1 : Approximations VS Solution 

plt.subplot(331)
plt.plot(t_vals[0], eu_y_vals[0], 'g.-', label='Euler Approximation')
plt.plot(t_vals[0], rk4_y_vals[0], 'b.-', label='RK4 Approximation')
plt.plot(t_vals[0], [sols[0](t) for t in t_vals[0]], 'r', label=r'$y=4+8t-5t^2$')
plt.title('Solution-Approximation Comparison')
plt.legend()

plt.subplot(334)
plt.plot(t_vals[1], eu_y_vals[1], 'g.-', label='Euler Approximation')
plt.plot(t_vals[1], rk4_y_vals[1], 'b.-', label='RK4 Approximation')
plt.plot(t_vals[1], [sols[1](t) for t in t_vals[1]], 'r', label=r'$ln(t^2-4t+e^{4})$')
plt.legend()

plt.subplot(337)
plt.plot(t_vals[2], eu_y_vals[2], 'g.-', label='Euler Approximation')
plt.plot(t_vals[2], rk4_y_vals[2], 'b.-', label='RK4 Approximation')
plt.plot(t_vals[2], [sols[2](t) for t in t_vals[2]], 'r', label=r'$4e^{(3/2)t^2}$')
plt.legend()


# Column 2 : Error graphs 

plt.subplot(332)
plt.plot(t_vals[0], eu_error_vals[0], 'purple', label='Euler Approximation')
plt.plot(t_vals[0], rk4_error_vals[0], 'orange', label='RK4 Approximation')
plt.title('Truncation Error Growth')
plt.legend()

plt.subplot(335)
plt.plot(t_vals[1], eu_error_vals[1], 'purple', label='Euler Approximation')
plt.plot(t_vals[1], rk4_error_vals[1], 'orange', label='RK4 Approximation')
plt.legend()

plt.subplot(338)
plt.plot(t_vals[2], eu_error_vals[2], 'purple', label='Euler Approximation')
plt.plot(t_vals[2], rk4_error_vals[2], 'orange', label='RK4 Approximation')
plt.legend()

# Column 3 : Vector fields for each ODE 

plt.subplot(333)
slope_field(fn_1)
plt.title('Vector Field')

plt.subplot(336)
slope_field(fn_2)

plt.subplot(339)
slope_field(fn_3)


plt.show()

