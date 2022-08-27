from kedro.framework.project import configure_project
from pathlib import Path
from cli import run
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import argparse

def main(project_root, env):    

    #project_root = "/home/ugupta/dev/code/sof-kedro-template"        
    bootstrap_project(project_root)    
    configure_project(Path(__file__).parent.name)

    with KedroSession.create(project_path=project_root, save_on_close=True, env=env) as session:
        session.run()

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Run Kedro Project')
    parser.add_argument('--root', metavar='path', required=True, help='path to project root folder')
    parser.add_argument('--env', metavar='conf', required=True, help='environment config')
    args = parser.parse_args()

    # call main
    main(project_root = args.root, env = args.env)