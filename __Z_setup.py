from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Education',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.8'
]

setup(
    name='ita-domes',
    version='0.0.1',
    author='betalab group UNIPD',
    author_email=['jacopo.vivian@hotmail.com','enrico.prataviera@unipd.it'],
    packages=find_packages(),
    scripts=[],
    url='https://github.com/BETALAB-team/ita-domes',
    license='LICENSE',
    description='A simulation tool for the evaluation of the energy consumption of Italian residential buildings.',
    long_description=open('README.md').read(),
    install_requires=[
	"pandas",
	"numpy",
	"pvlib",
	"scipy",
	"matplotlib",
	"geopandas",
	"pathlib",
	"pythermalcomfort",
	"blosc",
	"progressbar2",
    ],
    # extras_require={
    #        "dev":'matplotlib',
    #},
)
