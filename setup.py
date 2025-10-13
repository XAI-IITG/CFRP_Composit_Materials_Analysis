from setuptools import find_packages, setup

# Create properly formatted project name for package
PROJECT_NAME_NORMALIZED=$(echo "CFRP_Analysis" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

setup(
    name='${PROJECT_NAME_NORMALIZED}_src', # Replace with your desired package name for src
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='Source code for ${PROJECT_NAME_PARAM}',
    author='Your Name',
    author_email='your.email@example.com',
    # install_requires=[
    #    # List your project's dependencies here,
    #    # often read from requirements.txt
    # ],
    # entry_points={
    #    'console_scripts': [
    #        'my-ml-cli=${PROJECT_NAME_NORMALIZED}_src.application.cli.main:app',
    #    ],
    # },
)

