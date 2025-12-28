import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_test, y_prob, save_path="models/roc_curve.png"):

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — PD Model")
    plt.legend()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"ROC-кривая сохранена: {save_path}")
