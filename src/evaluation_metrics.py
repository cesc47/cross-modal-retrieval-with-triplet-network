import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from operator import truediv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
import umap
import cv2


def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def plot_confusion_matrix(ground_truth, predicted, show=False):
    """
    Plot the confusion matrix. MUST BE MAP1
    Parameters
    ----------
    ground_truth : list
                   A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # convert ground truth from a list of lists to a list
    ground_truth = [item for sublist in ground_truth for item in sublist]
    # convert predicted from a list of lists to a list
    predicted = [item for sublist in predicted for item in sublist]
    # compute the confusion matrix
    cm = confusion_matrix(ground_truth, predicted)

    if show:
        plt.figure(figsize=(9, 9))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = ['forest', 'opencountry', 'tallbuilding', 'mountain', 'street', 'insidecity', 'coast', 'highway']
        plt.xticks(np.arange(8), tick_marks, rotation=45)
        plt.yticks(np.arange(8), tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return cm


def table_precision_recall(cm, show=False):
    # compute the precision-recall curve
    tp = np.diag(cm)
    prec = list(map(truediv, tp, np.sum(cm, axis=0)))
    rec = list(map(truediv, tp, np.sum(cm, axis=1)))

    # for prec and rec compute the round to 4 decimal places
    prec = [round(x, 4) for x in prec]
    rec = [round(x, 4) for x in rec]
    if show:
        fig, ax = plt.subplots(1, 1)
        data = [prec,
                rec]
        column_labels = ['forest', 'opencountry', 'tallbuilding', 'mountain', 'street', 'insidecity', 'coast',
                         'highway']
        df = pd.DataFrame(data, columns=column_labels)
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=["Precision", "Recall"],
                 rowColours=["yellow"] * 8,
                 colColours=["yellow"] * 8,
                 loc="center",
                 fontSize=100)
        plt.show()

    return prec, rec

dict_classes = {'0': 'Coast',
                '1': 'Forest',
                '2': 'Highway',
                '3': 'Inside City',
                '4': 'Mountain',
                '5': 'Open Country',
                '6': 'Street',
                '7': 'Tall Building'}
def image_representation(features, classes, type='tsne'):
    """
    Plot the image representation.
    Parameters
    ----------
    features : list
               A list of features (order doesn't matter)
    classes : list
              A list of classes (order does matter)
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # Create the color palette
    color_palette = [c for c in list(mcolors.TABLEAU_COLORS)[:len(classes)]]

    # --- PCA ---
    if type == 'pca':
        # Create the PCA instance
        pca = PCA(n_components=2)
        pca.fit(features)
        pca_features = pca.transform(features)
        labels = np.unique(classes)

        for idx, label in enumerate(labels):
            label_features = [pca_features[i] for i, x in enumerate(classes) if x == label]
            plt.scatter(
                [x[0] for x in label_features],
                [x[1] for x in label_features],
                c=color_palette[idx],
                label=f'{idx}-{dict_classes[str(label)]}',
            )
        plt.legend(loc='best')

        for idx, label in enumerate(labels):
            label_features = [pca_features[i] for i, x in enumerate(classes) if x == label]
            xtext, ytext = np.median(label_features, axis=0)
            txt = plt.text(xtext, ytext, dict_classes[str(idx)])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])

        plt.title('2D PCA representation of the image features')
        plt.show()

    # --- TSNE ---
    elif type == 'tsne':
        # compute the t-SNE image representation
        tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca')
        tsne_features = tsne.fit_transform(features)
        # tsne_features = tsne_features.tolist()
        labels = np.unique(classes)

        # for feature, c in zip(tsne_features, labels):
        #     plt.scatter(feature[0], feature[1], c=color_map[c])

        for idx, label in enumerate(labels):
            label_features = [tsne_features[i] for i, x in enumerate(classes) if x == label]
            plt.scatter(
                [x[0] for x in label_features],
                [x[1] for x in label_features],
                c=color_palette[idx],
                label=f'{idx}-{label}',
            )

        plt.legend(loc='best')

        for idx, label in enumerate(labels):
            label_features = [tsne_features[i] for i, x in enumerate(classes) if x == label]
            xtext, ytext = np.median(label_features, axis=0)
            txt = plt.text(xtext, ytext, str(idx))
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])

        plt.title('2D TSNE representation of the image features')
        plt.show()

    # --- UMAP ---
    elif type == 'umap':
        # compute the umap image representation
        embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(features)
        labels = np.unique(classes)

        for idx, label in enumerate(labels):
            label_features = [embedding[i] for i, x in enumerate(classes) if x == label]
            plt.scatter(
                [x[0] for x in label_features],
                [x[1] for x in label_features],
                c=color_palette[idx],
                label=f'{idx}-{dict_classes[str(label)]}',
            )

        plt.legend(loc='best')

        for idx, label in enumerate(labels):
            label_features = [embedding[i] for i, x in enumerate(classes) if x == label]
            xtext, ytext = np.median(label_features, axis=0)
            txt = plt.text(xtext, ytext, dict_classes[str(idx)])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])

        plt.title('2D UMAP representation of the image features')
        plt.show()


def plot_prec_recall_map_k(type=None, **lists_k):
    for model, values in lists_k.items():
        k = np.arange(1, len(values) + 1)
        plt.plot(k, values, label=model, linewidth=2, marker='o')
    if type == 'precision':
        plt.title('Precision@k')
    if type == 'recall':
        plt.title('Recall@k')
    if type == 'mapk':
        plt.title('mapk@k')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel(type)
    plt.legend()
    plt.show()


def plot_image_retrievals(queries, retrivals, k=5):
    random_queries = np.random.randint(len(queries), size=k)

    fig, ax = plt.subplots(k, len(retrivals[0]) + 1)
    for i, query_num in enumerate(random_queries):
        query_path = queries[query_num]
        retrieval_paths = retrivals[query_num]
        query_image = cv2.imread(query_path)[:, :, ::-1]
        ax[i, 0].imshow(query_image)
        ax[i, 0].set_title(query_path.split('/')[-1].split("\\")[0])
        ax[i, 0].axis('off')
        for j, retrieval_path in enumerate(retrieval_paths):
            retrieval_image = cv2.imread(retrieval_path)[:, :, ::-1]
            ax[i, j + 1].imshow(retrieval_image)
            ax[i, j + 1].set_title(retrieval_path.split('/')[-1].split("\\")[0])
            ax[i, j + 1].axis('off')
    plt.show()
