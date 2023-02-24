# All in
from matplotlib import pyplot as plt
import numpy as np

def ratio_function(muscle_gained):
    x = muscle_gained
    y_max = .55
    y_min = .20
    x_max = 20
    y = (((y_min - y_max) / (x_max ** 2)) * (x**2) + y_max) * np.exp(-(x**2)/400) + .2 + 0.075 * np.exp(-(1/100)*(x - 28)**2) + 0.01 * np.exp(-(1/100)*(x - 41)**2)
    return y

def ratio_function2(muscle_gained):
    return .2 + 0.55 * np.exp(-(muscle_gained**2) / 49)

def transformation(starting_weight, starting_bf_percent, muscle_gained, week):
    # Weight Gain
    fat_free_list = []
    fat_list = []
    fat_free_list.append(starting_weight * (1 - starting_bf_percent))
    fat_list.append(starting_weight * starting_bf_percent)
    bfp = starting_bf_percent
    weight = starting_weight
    weight_per_week = 1
    muscle_gained = 0
    while bfp < .16:
        ratio = ratio_function(muscle_gained)
        weight += weight_per_week
        bf = weight * bfp + (1 - ratio) * weight_per_week
        bfp = bf / weight
        print(bfp, weight, 'Week:', week) 
        muscle_gained += ratio * weight_per_week

        # Append Lists
        fat_free_list.append(weight * (1 - bfp))
        fat_list.append(weight * bfp)
        week += 1

    cut_weight_per_week = 1
    cut_ratio = 0.15
    while bfp > .13:
        weight -= cut_weight_per_week
        bf = weight * bfp - (1 - cut_ratio) * cut_weight_per_week
        bfp = bf / weight
        print(bfp, weight, 'Week:', week)
        week += 1
        muscle_gained -= cut_ratio * cut_weight_per_week
        
        # Append Lists
        fat_free_list.append(weight * (1 - bfp))
        fat_list.append(weight * bfp)

    print(muscle_gained)
    return weight, bfp, muscle_gained, week, fat_free_list, fat_list


ff_list = []
f_list = []

weight2, bfp2, muscle_gained2, week2, f1, f2 = transformation(200, .13, 0, 0)
ff_list += f1
f_list += f2
weight3, bfp3, muscle_gained3, week3, f1, f2 = transformation(weight2, bfp2, muscle_gained2, week2)
ff_list += f1
f_list += f2
weight4, bfp4, muscle_gained4, week4, f1, f2 = transformation(weight3, bfp3, muscle_gained3, week3)
ff_list += f1
f_list += f2
weight5, bfp5, muscle_gained5, week5, f1, f2 = transformation(weight4, bfp4, muscle_gained4, week4)
ff_list += f1
f_list += f2
weight6, bfp6, muscle_gained6, week6, f1, f2 = transformation(weight5, bfp5, muscle_gained5, week5)
ff_list += f1
f_list += f2

print(weight6, bfp6, muscle_gained6, week6)
total = []
for i in range(len(ff_list)):
    total.append(ff_list[i] + f_list[i])

plt.figure()
plt.plot(f_list)
plt.plot(ff_list)
plt.plot(total)
plt.legend(['Fat Mass', 'Fat Free Mass', 'Total Mass'])
plt.xlabel('Time (Weeks)')
plt.ylabel('Weight (Pounds)')
plt.show()
