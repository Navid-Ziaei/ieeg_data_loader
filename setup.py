from setuptools import setup, find_packages

setup(
    name="ieeg_data_loader",
    version="0.1.2",
    author="author",
    author_email="nziaei@wpi.edu",
    description="A package for loading iEEG dataset",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Navid-Ziaei/ieeg_data_loader",
    install_requires=[
        'bokeh>=3.3.0',
        'colorcet>=3.0.1',
        'h5py>=2.10.0',
        'holoviews>=1.18.1',
        'joblib>=1.1.1',
        'matplotlib>=3.6.1',
        'mne>=1.2.1',
        'mne_bids>=0.12',
        'networkx>=2.7',
        'numpy>=1.23.5',
        'pandas>=1.5.1',
        'plotly>=5.15.0',
        'scikit_learn>=1.1.2',
        'scipy>=1.9.1',
        'seaborn>=0.10.1',
        'setuptools>=65.6.3',
        'tqdm>=4.64.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6"
)