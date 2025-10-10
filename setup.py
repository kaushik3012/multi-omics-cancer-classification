from setuptools import setup, find_packages


install_packages = [
    'captum>=0.4.1',
    'mygene>=3.2.2',
    'openpyxl>=3.0.9',
    'packaging>=21.3',
    'pandas>=1.2.5',
    'pandocfilters>=1.5.0',
    'seaborn>=0.11.2',
    'torch>=2',
    'scikit-learn>=1.2.2',
    'numpy>=1.23.5',
    'matplotlib>=3.6.2',
    'xgboost>=1.7.4',
    'livelossplot', 
    'tensorboardX',
    'tqdm',
    'ipython',
]

setup(
    name='AttentionMOI',
    author='',
    author_email='',
    version='0.1.2',
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=install_packages,
    dependency_links=[
        "https://pypi.org/simple/",
        "https://download.pytorch.org/whl/cpu#egg=torch",
        ],
    url='https://github.com/BioAI-kits/AttentionMOI',
    description="A Denoised Multi-omics Integration Framework for Cancer Subtype Classification and Survival Prediction.",
    license='Apache License 2.0',
    data_files=['AttentionMOI/example/cnv.csv.gz', 'AttentionMOI/example/met.csv.gz', 'AttentionMOI/example/rna.csv.gz', 'AttentionMOI/example/label.csv'],
    entry_points={
        'console_scripts': ['moi = AttentionMOI.moi:run_main',
                            ],
    },
)


