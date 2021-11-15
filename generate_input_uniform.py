from random import uniform

N = int(1e7)

fp = open("dataset_10m.txt", "w")

fp.write(f"{N}\n")
for i in range(N):
    fp.write(f"{uniform(-100, 100)} {uniform(-100, 100)}\n")
fp.close()

