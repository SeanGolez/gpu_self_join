import subprocess
import os
import pandas as pd

EPSILONS = ["0.001", "0.002", "0.003"]
DATASET = "/home/sgolez/datasets/iono_57min_5.16Mpts_3D_normalized_0_1.txt"
NUM_TRIALS = 3

def main():
    out_df = pd.DataFrame(columns=["Dataset", "Epsilon", "# random indexes", "Total execution time", "Kernel execution time", "Point comparisons", "Cell evaluations", "Total neighbors", "Method"])

    # loop through each option
    for option_idx in range(3):
        # clear output file
        open('py_test_stats.txt', 'w').close()

        # set the option
        # change the random index in parameters
        with open('params.h', 'r') as f:
            lines = f.readlines()
            if option_idx == 0:
                option = "RANDOMOFFSETSAMEALLDIM"
                lines[35] = "#define RANDOMOFFSETSAMEALLDIM 1\n"
                lines[36] = "#define FIXEDOFFSETALLDIM 0\n"
                lines[37] = "#define RANDOMOFFSETFOREACHDIM 0\n"
            if option_idx == 1:
                option = "FIXEDOFFSETALLDIM"
                lines[35] = "#define RANDOMOFFSETSAMEALLDIM 0\n"
                lines[36] = "#define FIXEDOFFSETALLDIM 1\n"
                lines[37] = "#define RANDOMOFFSETFOREACHDIM 0\n"
            if option_idx == 2:
                option = "RANDOMOFFSETFOREACHDIM"
                lines[35] = "#define RANDOMOFFSETSAMEALLDIM 0\n"
                lines[36] = "#define FIXEDOFFSETALLDIM 0\n"
                lines[37] = "#define RANDOMOFFSETFOREACHDIM 1\n"     

        with open('params.h', 'w') as f:
            f.writelines(lines)

        # loop throuch each random index 1-10
        for rand_idx in range(1,11):
            run_for_num_rand_indexes(rand_idx)

        # after 10, increment by powers of 2
        for rand_idx in ((10 + 2**p) for p in range(1, 7)):
            run_for_num_rand_indexes(rand_idx)
        
        new_df = pd.read_csv("py_test_stats.txt", names=["Dataset", "Epsilon", "#random indexes", "Total execution time", "Kernel execution time", "Point comparisons", "Cell evaluations", "Total neighbors"])
        new_df["Method"] = option

        out_df = pd.concat([out_df, new_df], ignore_index=True)

    out_df.to_csv("all_stats.csv", index=False)

    
def run_for_num_rand_indexes(rand_idx):
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


if __name__ == "__main__":
    main()