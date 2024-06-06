from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


def get_api_credentials():
    """Get the API endpoint and token for the current notebook context."""
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)
    api_root = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    )
    api_token = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
    return api_root, api_token
