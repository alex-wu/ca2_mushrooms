# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc,
    PrecisionRecallDisplay, precision_recall_curve,
    precision_score, recall_score
)
from sklearn.tree import DecisionTreeClassifier
import os  # Import os module for path operations

# Main function to run the app
def main():
    # Set the title and description of the app
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Predict whether mushrooms are **edible** or **poisonous** 🍄")
    st.sidebar.markdown("Predict whether mushrooms are **edible** or **poisonous** 🍄")

    # Load and preprocess data
    @st.cache_data(persist=True)
    def load_data():
        """
        Load the mushroom dataset and encode categorical variables.

        Returns:
            data (DataFrame): Preprocessed mushroom dataset.
        """
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the full path to 'Mushrooms.csv' in the same directory
            data_path = os.path.join(script_dir, "Mushrooms.csv")
            # Read the dataset
            data = pd.read_csv(data_path)
            # Initialize LabelEncoder
            label = LabelEncoder()
            # Encode each column
            for col in data.columns:
                data[col] = label.fit_transform(data[col])
            return data
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure 'Mushrooms.csv' is in the same directory as 'app.py'.")
            return None

    @st.cache_data(persist=True)
    def split_data(df):
        """
        Split the dataset into training and testing sets.

        Args:
            df (DataFrame): The preprocessed dataset.

        Returns:
            x_train, x_test, y_train, y_test: Split datasets.
        """
        y = df['type'].values  # Target variable
        x = df.drop(columns=['type']).values  # Features
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        """
        Plot selected performance metrics.

        Args:
            metrics_list (list): List of metrics to plot.
        """
        # Confusion Matrix
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            y_pred = model.predict(x_test)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=class_names, ax=ax)
            st.pyplot(fig)

        # ROC Curve
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            y_score = get_prediction_score(model, x_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc='lower right')
            st.pyplot(fig)

        # Precision-Recall Curve
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            y_score = get_prediction_score(model, x_test)
            precision, recall, _ = precision_recall_curve(y_test, y_score)

            fig, ax = plt.subplots()
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot(ax=ax)
            ax.set_title('Precision-Recall Curve')
            st.pyplot(fig)

    def get_prediction_score(model, x_test):
        """
        Get prediction scores for ROC and Precision-Recall curves.

        Args:
            model: Trained model.
            x_test: Test features.

        Returns:
            y_score: Prediction scores.
        """
        # Use predict_proba or decision_function depending on the model
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(x_test)
        else:
            y_score = model.predict(x_test)
        return y_score

    # Load data
    df = load_data()
    if df is not None:
        # Split data
        x_train, x_test, y_train, y_test = split_data(df)
        class_names = ['edible', 'poisonous']

        # Sidebar - Classifier selection
        st.sidebar.subheader("Choose Classifier")
        classifier_name = st.sidebar.selectbox(
            "Classifier", ("Support Vector Machine (SVM)",
                           "Logistic Regression",
                           "Random Forest",
                           "Decision Tree"))

        # Function to get classifier parameters
        def get_classifier_params(classifier_name):
            """
            Get hyperparameters for the selected classifier.

            Args:
                classifier_name (str): Name of the classifier.

            Returns:
                params (dict): Dictionary of hyperparameters.
            """
            params = dict()
            if classifier_name == 'Support Vector Machine (SVM)':
                params['C'] = st.sidebar.number_input(
                    "C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0)
                params['kernel'] = st.sidebar.radio("Kernel", ("rbf", "linear"), index=0)
                params['gamma'] = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), index=0)
            elif classifier_name == 'Logistic Regression':
                params['C'] = st.sidebar.number_input(
                    "C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0)
                params['max_iter'] = st.sidebar.slider(
                    "Maximum number of iterations", 100, 500, value=100)
            elif classifier_name == 'Random Forest':
                params['n_estimators'] = st.sidebar.number_input(
                    "Number of trees in the forest", 100, 5000, step=10, value=100)
                params['max_depth'] = st.sidebar.number_input(
                    "Maximum depth of the tree", 1, 20, step=1, value=5)
                params['bootstrap'] = st.sidebar.radio(
                    "Bootstrap samples when building trees", ('True', 'False'), index=0) == 'True'
            elif classifier_name == 'Decision Tree':
                params['max_depth'] = st.sidebar.number_input(
                    "Maximum depth of the tree", 1, 20, step=1, value=5)
                params['min_samples_split'] = st.sidebar.slider(
                    "Minimum number of samples required to split an internal node", 2, 10, value=2)
            return params

        # Get classifier hyperparameters
        params = get_classifier_params(classifier_name)

        # Function to get the classifier model
        def get_classifier(classifier_name, params):
            """
            Initialize the classifier with the given hyperparameters.

            Args:
                classifier_name (str): Name of the classifier.
                params (dict): Hyperparameters.

            Returns:
                model: Initialized classifier.
            """
            if classifier_name == 'Support Vector Machine (SVM)':
                model = SVC(C=params['C'], kernel=params['kernel'],
                            gamma=params['gamma'], probability=True)
            elif classifier_name == 'Logistic Regression':
                model = LogisticRegression(C=params['C'], max_iter=params['max_iter'])
            elif classifier_name == 'Random Forest':
                model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                               max_depth=params['max_depth'],
                                               bootstrap=params['bootstrap'],
                                               n_jobs=-1)
            elif classifier_name == 'Decision Tree':
                model = DecisionTreeClassifier(max_depth=params['max_depth'],
                                               min_samples_split=params['min_samples_split'])
            else:
                st.error("Unsupported classifier.")
                model = None
            return model

        # Initialize the classifier
        model = get_classifier(classifier_name, params)

        # Sidebar - Metrics selection
        st.sidebar.subheader("Choose Metrics to Plot")
        metrics = st.sidebar.multiselect(
            "Metrics", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        # Button to start classification
        if st.sidebar.button("Classify"):
            # Display results
            st.subheader(f"{classifier_name} Results")
            with st.spinner('Training the model...'):
                model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            # Calculate precision and recall
            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)
            # Display metrics
            st.write(f"Accuracy: **{accuracy:.2f}**")
            st.write(f"Precision: **{precision}**")
            st.write(f"Recall: **{recall}**")
            # Plot selected metrics
            plot_metrics(metrics)

        # Show raw data if checkbox is selected
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Mushroom Dataset (Classification)")
            st.write(df)
    else:
        st.stop()

# Run the app
if __name__ == '__main__':
    main()
