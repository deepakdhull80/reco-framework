import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def extract_metrics(log_file_path):
    """
    Reads a model training log file and extracts specified metrics (HR, NDCG, 
    Learning Rate, Train AVGLoss, Eval AVGLoss) across epochs.
    The use of re.search() ensures patterns are found anywhere in the line.
    
    Args:
        log_file_path (str): The path to the log file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the extracted metrics, or None on failure.
    """
    
    data_points = []
    
    # Variables to hold the latest metrics found for the current epoch block
    current_epoch = None
    latest_hr = None
    latest_ndcg = None
    latest_lr = None
    latest_train_loss = None
    latest_eval_loss = None

    # Regular Expressions for Pattern Matching
    hr_ndcg_pattern = re.compile(
        r"Eval HR: (\d+\.?\d*), NDCG: (\d+\.?\d*)"
    )
    
    lr_pattern = re.compile(
        r"Current Learning Rate: (\d+\.?\d*(?:e[+-]?\d+)?)" 
    )
    
    # Must contain "FINAL" and capture Epoch
    train_loss_pattern = re.compile(
        r"\[TRAIN\], Epoch: (\d+), FINAL, AVGLoss: (\d+\.?\d*)"
    )

    eval_loss_pattern = re.compile(
        r"\[EVAL\], Epoch: (\d+), FINAL, AVGLoss: (\d+\.?\d*)"
    )

    # File Processing and Data Extraction
    print(f"Reading log file: {log_file_path}...")
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                
                # A. Check for Performance Metrics (HR, NDCG, LR)
                
                match_hr_ndcg = hr_ndcg_pattern.search(line) 
                if match_hr_ndcg:
                    latest_hr = float(match_hr_ndcg.group(1))
                    latest_ndcg = float(match_hr_ndcg.group(2))
                    continue

                match_lr = lr_pattern.search(line)
                if match_lr:
                    latest_lr = float(match_lr.group(1))
                    continue

                # B. Check for Loss Metrics (AVGLoss and Epoch)
                
                match_train = train_loss_pattern.search(line)
                if match_train:
                    current_epoch = int(match_train.group(1))
                    latest_train_loss = float(match_train.group(2))
                    
                match_eval = eval_loss_pattern.search(line)
                if match_eval:
                    # Finalize the record when the EVAL FINAL line is hit
                    current_epoch = int(match_eval.group(1))
                    latest_eval_loss = float(match_eval.group(2))

                    # C. Finalize Record and Store Data
                    if (current_epoch is not None and 
                        latest_hr is not None and 
                        latest_ndcg is not None and 
                        latest_train_loss is not None and
                        latest_eval_loss is not None):
                        
                        data_points.append({
                            'epoch': current_epoch,
                            'HR': latest_hr,
                            'NDCG': latest_ndcg,
                            'LR': latest_lr, 
                            'Train_Loss': latest_train_loss,
                            'Eval_Loss': latest_eval_loss,
                        })

                        # Reset metrics for the next epoch block
                        current_epoch = None
                        latest_hr = None
                        latest_ndcg = None
                        # Keep latest_lr
                        latest_train_loss = None
                        latest_eval_loss = None
                    
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        return None

    if not data_points:
        print("No complete metric data points were extracted from the log file.")
        return None

    df = pd.DataFrame(data_points).drop_duplicates(subset=['epoch'], keep='last')
    print(f"\nSuccessfully extracted {len(df)} epoch data points.")
    
    return df

def generate_output_files(df, output_folder):
    """
    Saves the extracted DataFrame to a CSV and generates plots to PNG files.
    """
    if df is None:
        return
        
    # 1. Ensure Output Folder Exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory confirmed: {output_folder}")

    # 2. Save Data to CSV
    csv_path = os.path.join(output_folder, 'training_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to CSV: {csv_path}")

    # 3. Generate and Save Plots

    # --- Plot 1: Performance Metrics (HR and NDCG) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['HR'], label='HR (Hit Ratio)', linestyle='-', color='tab:blue')
    plt.plot(df['epoch'], df['NDCG'], label='NDCG', linestyle='--', color='tab:orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Performance Metrics', fontsize=12)
    plt.title('Hit Ratio (HR) and NDCG vs. Epoch', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'performance_metrics_hr_ndcg.png'))
    plt.close()

    # --- Plot 2: Loss Metrics (Train and Eval AVGLoss) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['Train_Loss'], label='Train AVGLoss (FINAL)', linestyle='-', color='tab:red')
    plt.plot(df['epoch'], df['Eval_Loss'], label='Eval AVGLoss (FINAL)', linestyle='--', color='tab:green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training and Evaluation Loss vs. Epoch', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_metrics_train_eval.png'))
    plt.close()

    # --- Plot 3: Learning Rate ---
    if df['LR'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['LR'], label='Learning Rate', linestyle='-', color='tab:purple')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate vs. Epoch', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'learning_rate.png'))
        plt.close()
        print(f"Plot saved: learning_rate.png")
    else:
        print("Note: Learning Rate plot skipped as no LR data was extracted.")
    
    print("\n--- Output Files Generated Successfully ---")
    print(f"Check the folder: {os.path.abspath(output_folder)}")


def main():
    """
    Main function to handle command-line arguments and script execution.
    """
    parser = argparse.ArgumentParser(
        description="Extracts model training metrics (HR, NDCG, Loss, LR) from a log file and saves them to a CSV and PNG plots.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # ----------------------------------------------------------------------
    # UPDATED ARGPARSER CONFIGURATION (using flags and required=True)
    # ----------------------------------------------------------------------
    parser.add_argument(
        '-f',
        '--log_file_path', 
        type=str, 
        dest="log_file_path",
        required=True, # Explicitly makes this flagged argument mandatory
        help="Path to the model training log file (e.g., 'training_log.txt'). This is a mandatory argument."
    )
    parser.add_argument(
        '-o',
        '--output_folder', 
        dest="output_folder",
        type=str, 
        required=True, # Explicitly makes this flagged argument mandatory
        help="Path to the folder where output files (CSV, PNGs) will be saved (e.g., 'metrics_output'). This is a mandatory argument."
    )
    # ----------------------------------------------------------------------

    args = parser.parse_args()

    # Step 1: Extract Data
    metrics_df = extract_metrics(args.log_file_path)

    # Step 2: Generate Output Files
    generate_output_files(metrics_df, args.output_folder)


if __name__ == '__main__':
    main()