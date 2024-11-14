# setup.py
from setuptools import setup, find_packages
import yaml
import re


def convert_conda_to_pip_requirement(requirement):
    """Convert conda dependency syntax to pip syntax."""
    if not isinstance(requirement, str):
        return None

    # Skip python and pip as they shouldn't be in install_requires
    if requirement.startswith(('python', 'pip')):
        return None

    # Convert conda version specifiers to pip format
    # Handle different patterns:
    # pkg=1.0.0 -> pkg==1.0.0
    # pkg>=1.0.0 -> pkg>=1.0.0 (already correct)
    # pkg=1.0.* -> pkg~=1.0.0
    # pkg -> pkg (no version specified)
    pattern = re.compile(r'([a-zA-Z0-9-_]+)(?:\s*([=<>]+)\s*([0-9\w\.*]+))?')
    match = pattern.match(requirement)

    if not match:
        return None

    name, operator, version = match.groups()

    # Package name mappings from conda to pip
    PACKAGE_MAPPING = {
        'torch': 'pytorch',
        'opencv': 'opencv-python',
        'cv2': 'opencv-python'
    }

    # Convert package name if needed
    name = PACKAGE_MAPPING.get(name.lower(), name)
    if not operator or not version:
        return name

    # Convert operators
    if operator == '=':
        return f"{name}=={version}"
    elif operator == '>=':
        return f"{name}>={version}"
    elif operator == '<=':
        return f"{name}<={version}"
    elif version.endswith('.*'):
        # Convert wildcard to compatible release
        return f"{name}~={version[:-2]}.0"

    return f"{name}{operator}{version}"


def get_requirements():
    """Extract and convert requirements from environment.yml."""
    with open("environment.yml") as f:
        env = yaml.safe_load(f)

    requirements = []

    # Process regular dependencies
    for dep in env.get("dependencies", []):
        if isinstance(dep, dict):  # Skip pip section
            continue

        req = convert_conda_to_pip_requirement(dep)
        if req:
            requirements.append(req)

    return requirements


setup(
    name="soccernet-calibration",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
    python_requires=">=3.9",
)
