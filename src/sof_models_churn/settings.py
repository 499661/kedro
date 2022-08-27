"""Project settings."""
from sof_models_churn.context import ProjectContext
from sof_models_churn.hooks import ProjectHooks, ModelTrainingHooks, ModelValidatingHooks,\
                                   DataValidationHooks, TensorFlowHooks, TableHistoryHooks,\
                                   DuplicateRowRemovalHooks

# Instantiate and list your project hooks here
HOOKS = (ProjectHooks(), ModelTrainingHooks(), ModelValidatingHooks(), DataValidationHooks(),
         TensorFlowHooks(), TableHistoryHooks(), DuplicateRowRemovalHooks())

# List the installed plugins for which to disable auto-registry
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Define where to store data from a KedroSession. Defaults to BaseSessionStore.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore

# Define keyword arguments to be passed to `SESSION_STORE_CLASS` constructor
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Define custom context class. Defaults to `KedroContext`
CONTEXT_CLASS = ProjectContext

# Define the configuration folder. Defaults to `conf`
# CONF_ROOT = "conf"
