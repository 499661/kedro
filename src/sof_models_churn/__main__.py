"""sof-models_churn file for ensuring the package is executable
as `sof-models-churn` and `python -m sof_models_churn`
"""

from pathlib import Path
from kedro.framework.project import configure_project
from cli import run

def main():
    configure_project(Path(__file__).parent.name)
    run()

if __name__ == "__main__":
    main()
