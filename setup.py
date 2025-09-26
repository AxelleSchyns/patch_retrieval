from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1",
    packages=find_packages(),  # automatically finds all folders with __init__.py
    install_requires=[
        "pandas",
        "numpy",
        # add any other dependencies you need
    ],
    python_requires='>=3.8',
)
