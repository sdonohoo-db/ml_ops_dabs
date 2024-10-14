# Databricks notebook source
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    # Train a sklearn model on the iris dataset
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(max_depth=7)
    clf.fit(X, y)
    # Take the first row of the training dataset as the model input example.
    input_example = X.iloc[[0]]
    # Log the model and register it as a new version in UC.
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        # The signature is automatically inferred from the input example and its predicted output.
        input_example=input_example,
        registered_model_name="shovakeemian_dev.ml_ops_dabs.iris_model",
    )

# COMMAND ----------

print('updating to version8, again!')

# COMMAND ----------


