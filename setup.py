from setuptools import setup, find_packages

setup(
    name="batchgfn",
    version="0.0.0",
    description="",
    url="https://github.com/s-a-malik/batchgfn",
    author="Shreshth Malik",
    author_email="shreshth@robots.ox.ac.uk",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "click==8.0.4",
        "torch==1.12.1",
        "numpy==1.23.3",
        "scipy==1.9.1",
        "pandas==1.5.0",
        "gpytorch==1.9.0",
        "seaborn==0.12.0",
        "datasets==2.5.2",
        "evaluate==0.2.2",
        "matplotlib==3.6.0",
        "torchvision==0.13.1",
        "scikit-learn==1.1.2",
        "transformers==4.22.2",
        "torch_scatter==2.0.9",
        "pytorch-lightning==1.7.7",
        "gym[classic_control]",
        "gfn @ git+https://github.com/saleml/gfn.git",
        "wandb>=0.14.0",
        "batchbald_redux==2.0.5",
    ],
    entry_points={
        "console_scripts": ["batchgfn=batchgfn.main:cli"],
    },
)
