# -*- encoding: utf-8 -*-
import setuptools

install_requires = ["torch>=1.5.0", "transformers==4.3.3", "stanza>=1.2", "sympy>=1.6", "ray>=1.3.0", "nltk>=3.5", "gensim>=4.0.1", "word2number>=1.1","ray[tune]"]

classifiers = [
    'Operating System :: OS Independent', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8'
]
setup_requires = []

extras_require = {}

with open("PYPI.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='mwptoolkit',
    version='0.0.7',
    url="https://github.com/LYH-YF/MWPToolkit",
    author=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    packages=[package for package in setuptools.find_packages() if package.startswith('mwptoolkit')],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False
)
