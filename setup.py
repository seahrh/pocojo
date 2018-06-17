from setuptools import setup, find_packages
__version__ = '3.0'
setup(
    name="tia_pocojo",
    version=__version__,
    python_requires='>=3.6.0',
    install_requires=[
        'nltk>=3.2.5,<4',
        'numpy>=1.14.2,<2',
        'pandas>=0.22.0,<1',
        'scikit-learn>=0.19.1,<1',
        'scipy>=1.0.1,<2',
        'sgcharts-stringx',
        'sgcharts-timex'
    ],
    dependency_links=[
        'git+https://github.com/seahrh/sgcharts-stringx.git@master#egg=sgcharts-stringx-1.0.0',
        'git+https://github.com/seahrh/sgcharts-timex.git@master#egg=sgcharts-timex-1.0.0'
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    description='description',
    license='MIT'
)
