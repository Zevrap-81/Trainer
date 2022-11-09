from setuptools import find_packages, setup

setup(
    name='trainer',
    packages=find_packages(include=['trainer']),
    version='0.1.0',
    description="""A simple trainer class to handle training of a pytorch model.
                   It supports logging with tensorboard, checkpointing, visualisation and more""",
    author='Parvez Mohammed',
    license='MIT',
    install_requires=['torch', 'numpy', 'matplotlib', 'tqdm', 'tensorboard', 'class-resolver'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)