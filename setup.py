from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='libmoon',
    version='0.2.2',
    author='Xiaoyuan Zhang et al.',
    author_email='xzhang2523-c@my.cityu.edu.hk',
    description='LibMOON: A Gradient-based MultiObjective OptimizatioN Library in PyTorch',
    packages=find_packages(
        # include=['solver.gradient.epo_solver',
        #          'solver.gradient',
        #          'src.libmoon']
        # include=find_packages()
    ),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=['numpy==1.26.2',
                    'torch==2.2.1',
                    'matplotlib==3.8.3',
                    'tqdm==4.66.2',
                    'pymoo==0.6.1.1',
                    'cvxopt==1.3.2',
                    'cvxpy==1.4.2',
                    'ffmpeg-python',
                    'ffmpeg',
                    'scikit-learn'
                      ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)