from setuptools import setup, find_packages

long_description = '''
VSR (Video Super-Resolution) is a library to upscale and improve the quality of low resolution videos and images.
'''

setup(
    name='VSR',
    version='1.0.0',
    author='Oleksandr Zakharchenko',
    description='Video Super Resolution',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=['imageio', 'moviepy', 'imageio-ffmpeg', 'numpy', 'tensorflow==2.*', 'tqdm', 'pyaml', 'PyYAML'],
    extras_require={
        'gpu': ['tensorflow-gpu==2.*'],
        'dev': ['bumpversion==0.5.3'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
)
