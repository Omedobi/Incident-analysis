from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluationViz:
    """
    A class to visualize model evaluation metrics such as confusion matrices 
    and feature importances.
    """
    
    @staticmethod
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
            axes = [axes]  # Ensure axes is iterable for a single model case

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
    
    @staticmethod
    def plot_feature_importance(model_names, feature_importances):
        """
        Plots the feature importance for each model.

        Args:
            model_names (list): List of model names.
            feature_importances (list of pd.DataFrame): List of feature importance dataframes for each model.

        Returns:
            None
        """
        fig, axes = plt.subplots(1, len(model_names), figsize=(20, 7))

        for i, (model_name, feature_importance) in enumerate(zip(model_names, feature_importances)):
            sorted_feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
            ax = axes[i]
            sns.barplot(x='Importance', y='Features Name', data=sorted_feature_importance, ax=ax)
            ax.set_title(f'Feature Importance for {model_name} Model', fontsize=14)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')

        plt.tight_layout()
        plt.show()
