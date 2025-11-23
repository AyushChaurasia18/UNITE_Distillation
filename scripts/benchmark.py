import os
import itertools
import warnings
import pandas as pd
from tqdm import tqdm
from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ================= Configuration =================
DATASET = 'cptac_ov'
TASK_NAME = 'Immune_class'

# Paths
POOLED_ROOT = r"D:\Histopathology project\Pooled_Features" # Where script 5 saved data
RESULTS_DIR = r"D:\Histopathology project\Benchmark_Results"

# Models and Levels to test
MODEL_NAMES = ['MyStudentViT', 'uni_v2']
LEVELS = [
    "20x_256px_0px_overlap",
    "40x_512px_0px_overlap",
    "20x_512px_0px_overlap"
]

def main():
    # 1. Get Split Config
    print("Fetching PathoBench split configuration...")
    path_to_split, path_to_task_config = SplitFactory.from_hf('./benchmark_splits', DATASET, TASK_NAME)
    
    # 2. Main Experiment Loop
    for model, level in itertools.product(MODEL_NAMES, LEVELS):
        print(f"\n{'#'*60}")
        print(f"BENCHMARKING: {model} @ {level}")
        print(f"{'#'*60}")

        # Construct path to the pooled features for this specific combo
        # Matches structure from script 5: /Pooled_Features/level/model/
        pooled_features_path = os.path.join(POOLED_ROOT, level, model)
        
        if not os.path.exists(pooled_features_path):
            print(f"❌ Features not found at: {pooled_features_path}. Skipping.")
            continue

        # --- Experiment A: Linear Probing ---
        try:
            print(f"  👉 Running Linear Probe (BACC)...")
            save_path_lin = os.path.join(RESULTS_DIR, "LinProbe", model, level)
            
            exp_lin = ExperimentFactory.linprobe(
                split=path_to_split,
                task_config=path_to_task_config,
                pooled_embeddings_dir=pooled_features_path,
                saveto=save_path_lin,
                combine_slides_per_patient=True,
                cost=1,
                balanced=False
            )
            exp_lin.train()
            exp_lin.test()
            score_lin = exp_lin.report_results(metric='bacc')
            print(f"     🏆 Linear Probe BACC: {score_lin}")
        except Exception as e:
            print(f"     ⚠️ Linear Probe Failed: {e}")

        # --- Experiment B: Retrieval ---
        try:
            print(f"  👉 Running Retrieval (mAP)...")
            save_path_ret = os.path.join(RESULTS_DIR, "Retrieval", model, level)
            
            exp_ret = ExperimentFactory.retrieval(
                split=path_to_split,
                task_config=path_to_task_config,
                pooled_embeddings_dir=pooled_features_path,
                saveto=save_path_ret,
                combine_slides_per_patient=True,
                similarity='l2',
                centering=False
            )
            exp_ret.train()
            exp_ret.test()
            map1 = exp_ret.report_results(metric='mAP@1')
            map5 = exp_ret.report_results(metric='mAP@5')
            print(f"     🏆 Retrieval mAP@1: {map1} | mAP@5: {map5}")
        except Exception as e:
            print(f"     ⚠️ Retrieval Failed: {e}")

if __name__ == "__main__":
    main()