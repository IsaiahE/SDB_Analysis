import os

# Creates a list of all the paths to be used by matlab in creating the animations

for file in os.listdir(r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis'):
    if file[-4:] == '.txt':
        continue
    if file == 'Nolte Set':
        continue 
    #print(file)
    for subfile in os.listdir(r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + file):
        print('"' + file + '\\' + subfile + '\\"; ')

        