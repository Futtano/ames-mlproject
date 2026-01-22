from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def find_requirements(path: str) -> List[str]:
    """
    Returns requirements list from a file

    Parameters
        path: str   file path

    Returns
        requirements : List[str]   list of requirements
    """
    with open(path, 'r', encoding='utf-8') as f:
        requirements = f.readlines()
        requirements = [line.replace('\n', '') for line in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ames-mlproject',
    version='0.0.1',
    author='Daniele Loru',
    author_email='futhanos@gmail.com',
    packages=find_packages(),
    install_requires=find_requirements('requirements.txt')
)