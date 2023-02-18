from setuptools import setup, find_packages

setup(
    name='sagemaker-example',
    version='1.0',
    description='SageMaker MLFlow Example',
    author='George Githiri',
    author_email='georgegithiri002@gmail.com',
    packages=find_packages(exclude=('test', 'docs'))
)