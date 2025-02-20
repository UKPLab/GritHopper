from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="grithopper",
    version="0.0.1",
    author="Justus-Jonas Erker",
    author_email="justus-jonas.erker@tu-darmstadt.de",
    download_url="https://github.com/UKPLab/GritHopper/",
    description="GritHopper: Decomposition-Free Multi-Hop Dense Retrieval",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires="==3.9.19",
    install_requires=[
        'gritlm==1.0.2',
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="PyTorch NLP deep learning Multi-Hop Dense Retrieval Fact-Checking Question Answering",
)