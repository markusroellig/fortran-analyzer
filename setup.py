from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fortran-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing Fortran source code variable usage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fortran-analyzer",
    py_modules=["fortran_variable_analyzer"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Fortran",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "fortran-analyzer=fortran_variable_analyzer:main",
        ],
    },
)
