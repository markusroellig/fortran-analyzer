from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
def get_version():
    with open("fortran_analyzer/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="fortran-analyzer",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive tool for analyzing Fortran source code variable usage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markusroellig/fortran-analyzer",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Fortran",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[
        # No external dependencies currently
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "fortran-analyzer=fortran_analyzer:main",
            "fortran-variable-analyzer=fortran_analyzer:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/fortran-analyzer/issues",
        "Source": "https://github.com/yourusername/fortran-analyzer",
        "Documentation": "https://github.com/yourusername/fortran-analyzer/blob/main/README.md",
    },
)
