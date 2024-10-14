import subprocess
import os

EPSILONS = ["0.001", "0.002", "0.003"]
DATASET = "/home/sgolez/datasets/iono_57min_5.16Mpts_3D_normalized_0_1.txt"
NUM_TRIALS = 3

# loop throuch each random index 1-10
for rand_idx in range(1,11):
    # change the random index in parameters
    with open('params.h', 'r') as f:
        lines = f.readlines()
        lines[25] = f"#define NUMRANDINDEXES {rand_idx}\n"

    with open('params.h', 'w') as f:
        f.writelines(lines)

    # compile
    subprocess.run(["make"])

    # loop through each epsilon
    for epsilon in EPSILONS:
        # do this for num trials
        for trial in range(NUM_TRIALS):
            subprocess.run(["./main", DATASET, epsilon, "3"])
    