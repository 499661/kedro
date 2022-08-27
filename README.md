# sof-models-churn

## Overview

This is the repository for the customer churn project. The documentation for the
project can be found on the company Wiki:
https://wiki.owfg.com/pages/viewpage.action?pageId=175905203

Note that the project uses `Kedro 0.17.7`. You can find the documentation for
this version of kedro here https://kedro.readthedocs.io/en/0.17.7/ .


## Rules and guidelines

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation. This
project does not use conda, but if you need to use it declare the dependencies
in `src/environment.yml`.

To install the requirements using pip do:

```bash
pip install -r src/requirements.txt
```

Furthermore, for local development you need to have `databricks-connect`
installed. You need to install it after installing the dependencies listed in
`src/requirements.txt` and then configure it as described here
https://wiki.owfg.com/pages/viewpage.action?pageId=154845001 . When working on
this project you should be using the `churn-dev` cluster or in some cases the
`churn-staging` cluster. You can find information about these clusters on the
databricks web interface.

## How to run Kedro

There are 4 different kedro environemts:
- `dbconnect`: used for "local" development with databricks-connect.
- `dev`: used for development
- `stage`: uses a separate databricks workspace for staging.
- `prod`: uses a separate databricks workspace for production.

For all environments the pipelines need to be run in order as shown below. Note
that `<env>` needs to be replaced with the name of the environment that you want
to use.
```bash
kedro run --pipeline=de --env=<env> # data engineering pipeline
kedro run --pipeline=ptt --env=<env> # prepare train test data pipeline
kedro run --pipeline=mt --env=<env> # model training pipeline
kedro run --pipeline=mv --env=<env> # model validating pipeline
kedro run --pipeline=pdn --env=<env> # prediction pipeline
```
Note that by default the `dbconnect` environment is used.


Note that the following will not work:
```bash
kedro run --env=<env>
```
The reason for this is that some pipelines depend on other pipelines in a way
that is not obvious to kedro and hence some nodes may be executed in the wrong
order.

### Running kedro in the dbconnect env

1) In `conf/dbconnect/globals.yml` add your username in username field. `username: my_username` 
    eg: `username: rajat_paliwal`


2) Also make sure to start your local mlflow server using :
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /abs/path/to/databricks-sof-models-churn/mlruns --host localhost
    ```

    Alternatively you can use the provided script and do:
    ```bash
    ./start-mlflow-server.sh /abs/path/to/databricks-sof-models-churn/mlruns
    ```

3) In order to run the pipeline do:
    ```bash
    kedro run --env=dbconnect --pipeline="name_of_pipeline"
    ```
    It's best to run the pipelines in order i.e. first de -> ptt -> mt -> mv -> pdn

### Running kedro in the dev env and other envs.

1) In `conf/dev/globals.yml` add your username in username field. `username: my_username` 
    eg: `username: rajat_paliwal`

2) Go to `notebook/kedro-run-notebook`.

3) Make sure to attach the `kedro-run-notebook` to "churn-dev" cluster and run it.

Optionally: 

1). If you are planning to run the `churn-dev-run` job which runs the `kedro-run-notebook` make sure to go to the `workflows -> churn_dev_run -> Tasks` and
    then update the `Path` field to a location of your liking. By default it will be set to the location of kedro-run-notebook in the `/repos/shared/databricks-sof-models-churn/`  or somebody else's repos location.

2). Also make sure to set the `env` job parameter to `dev`.

3). For `churn_stage_run` in the staging space make sure to set the `env` job parameter to `stage`.

4). For `churn_prod_run` in the staging space make sure to set the `env` job parameter to `prod` and if you are doing the training as well make sure to set the `action` param to `train`.


## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, look at the `.coveragerc` file.

## How to visualize your kedro pipeline

Using kedro viz we can visualize all the sub pipelines involved in churn pipeline and see all the nodes along with their inputs and outputs. This really helps in understanding the project better.

```
kedro viz
```
Perform this command in your `dbconnect` env and you will be able to see visualizations as shown below:
`de pipeline`
![image](https://user-images.githubusercontent.com/98116675/185982428-da741358-ebac-4866-95dc-1b7b97e936e5.png)




