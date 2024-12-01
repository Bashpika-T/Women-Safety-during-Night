import pandas as pd

def calculate_metrics_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize lists to store metrics
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    # Initialize total TP, TN, FP, FN for overall metrics
    total_tp = total_tn = total_fp = total_fn = 0

    # Calculate metrics for each attribute
    for index, row in df.iterrows():
        tp = row['True Positive']
        tn = row['True Negative']
        fp = row['False Positive']
        fn = row['False Negative']
        
        # Calculate metrics for current attribute
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Append to lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        # Update total counts
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

    # Prepare results
    metrics = {
        'Attribute Metrics': {
            'Precision': precision_list,
            'Recall': recall_list,
            'F1 Score': f1_score_list
        },
        'Overall Metrics': {
            'Overall Precision': overall_precision,
            'Overall Recall': overall_recall,
            'Overall F1 Score': overall_f1_score,
            'Overall Accuracy': overall_accuracy
        }
    }

    return metrics

# Example usage
file_path_peta = 'confusion_matrices/peta_confusion_matrix.csv'
file_path_pa100k = 'confusion_matrices/pa100k_confusion_matrix.csv'

peta_metrics = calculate_metrics_from_csv(file_path_peta)
pa100k_metrics = calculate_metrics_from_csv(file_path_pa100k)

print("PETA Metrics:")
print(peta_metrics)
print("\nPA100K Metrics:")
print(pa100k_metrics)
