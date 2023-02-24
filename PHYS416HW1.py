import math
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort
# PHYS 416 HW1

def Quanta_Box():
    N1 = 300;
    N2 = 200;

    def Omega(q, N):
        return math.factorial(q + N - 1) / (math.factorial(q) * math.factorial(N - 1))

    data = [0] * 101
    data1 = [0] * 101
    data2 = [0] * 101

    qT = 100
    temp = 0
    m = 4.48e-26
    h_bar = 1.0545e-34
    kb = 1.38e-23

    temp1 = []
    temp2 = []

    for i in range(1, qT):
        data[i] = np.log(Omega(i, N1) * Omega(qT - i, N2))
        data1[i] = np.log(Omega(i, N1))
        data2[i] = np.log(Omega(qT - i, N2))
        ds1 = kb * (np.log(Omega(i, N1)) - np.log(Omega(i-1, N1)))
        ds2 = kb * (np.log(Omega(qT - i, N2)) - np.log(Omega(qT - i - 1, N2)))
        temp1 = temp1 + [h_bar*(8*math.sqrt(1/m))/ds1]
        temp2 = temp2 + [h_bar*(8*math.sqrt(1/m))/ds2]

    X = range(1, qT)

    print(max(data))
    print(data.index(max(data)))
    #for i in range(len(data)):
    #    if data[i] > .8 * max(data) / 2:
    #        if data[i] < 1.2 * max(data) / 2:
    #            print(i, data[i])


    plt.plot(X, temp1)
    plt.plot(X, temp2)
    plt.title('Temperature Plot')
    plt.xlabel('Quanta in Block 1')
    plt.ylabel('Temperature (K)')
    #plt.legend('ln(Omega1*Omgea2)', 'Ln(Omega1)', 'Ln(Omega2)')
    plt.show()


def Birthday():

    def BirthdayCoincidences(K, C):
        
        count = 0
        for classes in range(0, C):
            rando_list = []
            for i in range(0, K):
                rando_list.append(random.randint(0, 2^32))
            sroted = sort(rando_list)
            for j in range(1, len(sroted)):
                if sroted[j] - sroted[j-1] == 0:
                    count += 1
                    break
        
        return count / C

    kiddos = 90
    kiddo_list = [0] * kiddos 
    N = 1
    for what in range(0, N):
        for k in range(0, kiddos):
            print(k)
            kiddo_list[k] += BirthdayCoincidences(k*1000, 10) / N


    plt.plot(range(0, kiddos), kiddo_list)
    plt.title('Probability of having repeated birthdays')
    plt.xlabel('Number of kids in each class')
    plt.ylabel('Probability')
    plt.show()

print(0.49, 'At 75000')
print(0.50, 'At 75500')
print(0.47, 'At 76000')
print(0.49, 'At 76500')

Birthday()

print(0.49, 'At 75000')
print(0.50, 'At 75500')
print(0.47, 'At 76000')
print(0.49, 'At 76500')