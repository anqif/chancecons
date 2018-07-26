from setuptools import setup

setup(name='chancecons',
      version='0.1',
      description='A package that implements chance (quantile) constraints in CVXPY using a 2-step majorization-minimization algorithm.',
      url='http://github.com/anqif/chancecons',
      author='Anqi Fu, Stephen Boyd',
      author_email='anqif@stanford.edu',
      license='Apache License, Version 2.0',
      packages=['chancecons'],
      install_requires=['cvxpy >= 1.0',
						'numpy >= 1.14'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
