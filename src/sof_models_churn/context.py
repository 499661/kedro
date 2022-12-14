"""Entry point for running a Kedro pipeline as a Python package."""
from pathlib import Path
from typing import Any, Dict, Union

from kedro.framework.context import KedroContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import mlflow

class ProjectContext(KedroContext):
    """A subclass of KedroContext to add Spark initialisation for the pipeline."""

    def __init__(
        self,
        package_name: str,
        project_path: Union[Path, str],
        env: str,
        extra_params: Dict[str, Any] = None,
    ):
        # We do not have a local env and therefore make dbconnect the default
        if env is None:
            env = "dbconnect"

        super().__init__(package_name, project_path, env, extra_params)
        self.init_spark_session()
        self.init_mlflow()
        

    def init_spark_session(self) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = self.config_loader.get("spark*", "spark*/**")
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(self.package_name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")

    
    
    def init_mlflow(self) -> None:
        """Sets up Mlflow tracking URI and experiment name
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = self.config_loader.get("parameters*", "parameters*/**")
        mlflow.set_tracking_uri(parameters["mlflow"]["mlflow_tracking_uri"])
        mlflow.set_experiment(parameters["mlflow"]["mlflow_experiment"])