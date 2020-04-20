import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Abra",
    version="0.0.1",
    author="nbcd14",
    author_email="nbcd14@gmail.com",
    description="A package for linear model selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/nbcd14/Abra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)