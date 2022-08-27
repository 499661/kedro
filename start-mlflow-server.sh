#!/bin/bash

# This is a convenience script for starting the mlflow server.
# It is only required when using the dbconnect environment
if [ $1 = ""]; then
	echo "Error: you must pass a path for the default-artifact-root"
else
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $1 --host localhost
fi

