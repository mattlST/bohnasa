from setuptools import setup

setup(
    name='bohnasa',
    version='0.0.0',    
    description='A Python package for exact diagonalization tools in quantum many body systems',
    url='https://github.com/mattlST/bohnasa',
    author='Mathias Steinhuber',
    author_email='157615034+mattlST@users.noreply.github.com',
    packages=['bohnasa'],
    install_requires=['math',
                      'setuptools',
                      'numpy',
                      'scipy',
                      'numba',
                      'jax',    
                      'matplotlib',
                      'h5py'         
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)
