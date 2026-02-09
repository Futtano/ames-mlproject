from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def find_requirements(path: str) -> list[str]:
    """
    Returns requirements list from a file

    Parameters
        path: str   file path

    Returns
        requirements : List[str]   list of requirements
    """
    with open(path, encoding="utf-8") as f:
        requirements = f.readlines()
        requirements = [line.replace("\n", "") for line in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="ames-mlproject",
    version="0.0.1",
    author="Daniele Loru",
    author_email="futhanos@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=find_requirements("requirements.txt"),
)
