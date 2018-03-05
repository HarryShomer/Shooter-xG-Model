import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import clean_data_xg as clean_data


def get_roc(actual, predictions):
    fig = plt.figure()
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    for key in predictions.keys():
        # Convert preds to just prob of goal
        preds = [pred[1] for pred in predictions[key]]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, preds)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print("ROC: ", roc_auc)
        plt.plot(false_positive_rate, true_positive_rate, 'b')
        plt.text(.1, .9, "AUC="+str(round(roc_auc, 3)))

    fig.savefig("ROC_xG.png")


def fit_gradient_boosting(features_train, labels_train):
    """
    Fit a gradient boosting algorithm 

    :return: classifier
    """
    clf = GradientBoostingClassifier(n_estimators=500, learning_rate=.1, random_state=42, verbose=2,
                                     max_depth=5,
                                     min_samples_split=100
                                     )

    print("Fitting Gradient Boosting Classifier")
    clf.fit(features_train, labels_train)

    # Save model
    print("\nGradient Boosting Classifier:", clf)
    pickle.dump(clf, open("gbm_xg_shooter_fixed_pos.pkl", 'wb'))

    return clf


def xg_model():
    """
    Create and test xg model
    """
    data = pd.read_csv("shooter_xg_shuffled.csv", index_col=0)

    # Convert to lists (disregard foo)
    features, labels, foo = clean_data.convert_data(data)

    # Split into training and testing sets -> 80/20
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.2, random_state=42)

    # Fix Data
    features_train, labels_train = np.array(features_train), np.array(labels_train).ravel()

    # FIT MODEL
    gb_clf = fit_gradient_boosting(features_train, labels_train)

    # Testing
    gb_preds_probs = gb_clf.predict_proba(features_test)

    # Convert test labels to list instead of lists of lists
    flat_test_labels = [label[0] for label in labels_test]

    # LOG LOSS
    print("\nLog Loss: ", log_loss(flat_test_labels, gb_preds_probs))

    # ROC
    preds = {"Gradient Boosting": gb_preds_probs}
    get_roc(flat_test_labels, preds)


def main():
    xg_model()


if __name__ == '__main__':
    main()

