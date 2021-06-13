import os

file1 = open('url.txt', 'r')
Lines = file1.readlines()

now = int(1)
for line in Lines:
    filename = line.rstrip(line[-1])
    os.system("curl " + str(filename) + " > " + str(now) + ".png")
    now = now + 1
