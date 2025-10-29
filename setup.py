# sepsis_detection_system/setup.py
from setuptools import setup, find_packages
import os

setup(
    name="sepsis_detection_system",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
        "matplotlib",
        "seaborn",
        "shap",
        "torch",
        "plotly",
        "fastapi",
        "uvicorn",
        "pydantic",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Sepsis Detection System",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sepsis_detection_system",
)