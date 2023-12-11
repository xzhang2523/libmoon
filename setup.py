from setuptools import setup, find_packages

setup(
    name='mootorch',
    version='0.1.0',
    author='Xiaoyuan Zhang et al.',
    author_email='xzhang2523-c@my.cityu.edu.hk',
    description='Make MOO great again',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)