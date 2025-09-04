import os
import re
import pandas as pd

def parse_log_file(file_path):
    """Parses a single log file to extract key experimental results."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract details from the command line arguments
    dataset_match = re.search(r"--dataset (\S+)", content)
    seed_match = re.search(r"--seed (\d+)", content)
    llm_match = re.search(r"--use_llm_pruning", content)

    dataset = dataset_match.group(1) if dataset_match else 'N/A'
    seed = int(seed_match.group(1)) if seed_match else 'N/A'
    method = "LLM-Enhanced" if llm_match else "Baseline"

    # Extract the final best test accuracy
    acc_match = re.search(r"\(Retrain Stage\) best test acc: (\d+\.\d+)", content)
    best_acc = float(acc_match.group(1)) if acc_match else float('nan')

    return {
        "Dataset": dataset,
        "Method": method,
        "Seed": seed,
        "Best Test Accuracy": best_acc
    }

def main():
    """
    Walks through the experimental results directory, parses all log files,
    and compiles them into a single CSV and a formatted Markdown table.
    """
    results = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith(".log") and root != '.':
                try:
                    file_path = os.path.join(root, file)
                    results.append(parse_log_file(file_path))
                except Exception as e:
                    print(f"Error parsing file {file_path}: {e}")

    if not results:
        print("No log files found to parse.")
        return

    df = pd.DataFrame(results)
    
    # --- Save to CSV ---
    csv_path = 'results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to {csv_path}")

    # --- Create and print summary table ---
    summary = df.groupby(['Dataset', 'Method'])['Best Test Accuracy'].agg(['mean', 'std']).reset_index()
    summary['std'] = summary['std'].fillna(0) # Replace NaN with 0 for single-run cases
    summary['Mean Accuracy (± Std Dev)'] = summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    
    summary_pivot = summary.pivot(index='Dataset', columns='Method', values='Mean Accuracy (± Std Dev)')
    
    print("\n--- Experimental Results Summary ---")
    print(summary_pivot.to_markdown())
    
    # --- Save summary to a Markdown file ---
    md_path = 'results_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Experimental Results Summary\n\n")
        f.write(summary_pivot.to_markdown())
    print(f"\nSummary table saved to {md_path}")


if __name__ == "__main__":
    main()