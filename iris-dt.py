import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Initialize DagsHub tracking
dagshub.init(repo_owner='Shrijeet14', repo_name='mlflow-dagshub-demo', mlflow=True)

# Set tracking URI to DagsHub MLflow server
mlflow.set_tracking_uri("https://dagshub.com/Shrijeet14/mlflow-dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Decision Tree model
max_depth = 10 

try:
    # Create or set the experiment
    experiment_name = "iris-dt"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name='skv'):
        # Initialize and train the Decision Tree Classifier
        dt = DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(X_train, y_train)

        # Make predictions
        y_pred = dt.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('max_depth', max_depth)

        # Generate classification report
        clf_report = classification_report(y_test, y_pred, target_names=iris.target_names)
        mlflow.log_text(clf_report, "classification_report.txt")

        # Create a confusion matrix plot
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris.target_names, 
                    yticklabels=iris.target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Iris Dataset')
        
        # Save the plot as an artifact
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()  # Close the plot to free up memory

        # Log artifacts
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact(__file__)

        # Log the model
        mlflow.sklearn.log_model(dt, "Decision_Tree_Model")

        # Set tags
        mlflow.set_tag('author', 'skv')
        mlflow.set_tag('model', 'Decision Tree')
        mlflow.set_tag('dataset', 'Iris')

        print(f'Accuracy: {accuracy}')
        print(f'Classification Report:\n{clf_report}')

except Exception as e:
    print(f"An error occurred: {e}")