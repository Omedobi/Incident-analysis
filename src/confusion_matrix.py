from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_preds, model_names, class_labels=None):
    """
    Plot confusion matrices for multiple models.
    
    Args:
        y_true (array-like): True labels.
        y_preds (list of array-like): Predictions from different models.
        model_names (list): List of model names corresponding to y_preds.
        class_labels (list, optional): Class labels for the confusion matrix axes.
    
    Returns:
        None
    """
    num_models = len(y_preds)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    
    if num_models == 1:
        axes = [axes] 

    for i, (pred, model_name) in enumerate(zip(y_preds, model_names)):
        cm = confusion_matrix(y_true, pred)
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax, cbar=False)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        if class_labels:
            ax.xaxis.set_ticklabels(class_labels)
            ax.yaxis.set_ticklabels(class_labels)

    plt.tight_layout()
    plt.show()