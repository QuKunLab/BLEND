# setup.py
from setuptools import setup, find_packages

setup(
    name="blend",                           # 发布到 PyPI 的包名（小写更规范）
    version="0.1.0",                        # 版本号（建议采用语义化版本）
    description="BLEND: identifies functional domains from subcellular-resolution spatial transcriptomics data",
    license="MIT",
    #url="https://github.com/yourname/blend",  # 如无仓库可先留空或删掉本行

    # 自动发现包；假设你的代码在根目录下的 blend/ 里（非 src 布局）
    packages=find_packages(exclude=("tests", "docs")),

    # 运行所需的依赖
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "matplotlib",
        "scikit-learn",
        "h5py",
        "anndata",
        "scanpy",
        "jupyter",
        "jupyterlab",
        "ipython"
    ]
)
