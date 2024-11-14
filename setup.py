# setup.py
from setuptools import setup, find_packages
import yaml


def get_requirements():
    with open("environment.yml") as f:
        env = yaml.safe_load(f)

    # Get non-pip dependencies
    deps = [dep for dep in env["dependencies"]
            if isinstance(dep, str) and dep != "pip"]

    # Remove python dependency
    deps = [dep for dep in deps if not dep.startswith("python")]

    return deps


setup(
    name="soccernet-calibration",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
    python_requires=">=3.9",
)
