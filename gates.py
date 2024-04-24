import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def abline(slope, intercept):
    """Plot a line given slope and intercept."""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


#----------------------------------------
# AND GATE
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 2
print("AND")
print("x1    x2    w1   w2     t     O")
for i in range(len(x1)):
    if (x1[i] * w1[i] + x2[i] * w2[i]) >= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)

# Scatter plot for the inputs
df = pd.DataFrame({'x1': [0, 0, 1, 1],
                   'x2': [0, 1, 0, 1]})
plt.scatter(df.x1, df.x2)

# Plotting the decision boundary (line)
abline(-1, 2)
plt.show()  # Display the plot


#----------------------------------------
#OR GATE
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 1
print("\nOR")
print("x1    x2    w1   w2     t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)

df = pd.DataFrame({'x1' : [0, 0, 1, 1],
                   'x2' : [0, 1, 0, 1]})

plt.scatter(df.x1, df.x2)

abline(-1, 1)
plt.show()


#----------------------------------------
#NOT GATE
x = [0, 1]
w = [-1, -1]
t = 0
print("\nNOT")
print("x     w     t     O")
for i in range(len(x)):
    if ( x[i]*w[i] ) >= t:
        print(x[i],'  ',w[i],'   ',t,'   ', 1)
    else:
        print(x[i],'  ',w[i],'   ',t,'   ', 0)
df = pd.DataFrame({'x1' : [0, 0, 1, 1],
                   'x2' : [0, 1, 0, 1]})
plt.scatter(df.x1, df.x2)
abline(-1, 0)
plt.show()


#----------------------------------------
#NAND GATE
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [-1, -1, -1, -1]
w2 = [-1, -1, -1, -1]
t = -2
print("\nNAND")
print("x1    x2    w1     w2    t   O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) > t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],' ',t,' ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],' ',t,' ', 0)

df = pd.DataFrame({'x1' : [0, 0, 1, 1],
                   'x2' : [0, 1, 0, 1]})

plt.scatter(df.x1, df.x2)
abline(-1, -2)
plt.show()


#----------------------------------------
#NOR Gate
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [-1, -1, -1, -1]
w2 = [-1, -1, -1, -1]
t = -1
# Print header for the output table
print("x1    x2    w1     w2    t   O")
for i in range(len(x1)):
    if (x1[i] * w1[i] + x2[i] * w2[i]) > t:
        output = 1
    else:
        output = 0
    print(x1[i], '   ', x2[i], '   ', w1[i], '   ', w2[i], ' ', t, ' ', output)

#Graph Code
df = pd.DataFrame({'x1': [0, 0, 1, 1],
                   'x2': [0, 1, 0, 1]})
plt.scatter(df.x1, df.x2)
abline(1, 1)
plt.show()


#----------------------------------------
#XOR GATE

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
w3 = [1, 1, 1, 1]
w4 = [-1, -1, -1, -1]
w5 = [-1, -1, -1, -1]
w6 = [1, 1, 1, 1]
t1 = [0.5,0.5,0.5,0.5]
t2 = [-1.5,-1.5,-1.5,-1.5]
t3 = [1.5,1.5,1.5,1.5]
def XOR (a, b):
    if a != b:
        return 1
    else:
        return 0

print("\nXOR")
print('x1    x2    w1    w2    w3     w4    w5     w6    t1    t2    t3    O')
for i in range(len(x1)):
    print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',w3[i],'   ',w4[i],'   ',w5[i],'   ',w6[i],'   ',t1[i],'   ',t2[i],'   ',t3[i],'   ',XOR(x1[i],x2[i]))

df = pd.DataFrame({'x1' : [0, 0, 1, 1],
                   'x2' : [0, 1, 0, 1]})

plt.scatter(df.x1, df.x2, color=['red','green','green','black'])

abline(-1, 1.5)

abline(-1, 0.5)
plt.show()
