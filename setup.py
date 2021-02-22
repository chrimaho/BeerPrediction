from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A trained custom neural networks model that will accurately predict a type of beer based on some rating criterias such as appearance, aroma, palate or taste. It is also built as a web app and deployed online in order to serve your model for real time predictions.',
    author='Chris Mahoney',
    license='MIT',
)
