from distutils.core import setup
import setuptools

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename) as f:
        required = f.read().splitlines()
    return required


setup(name='challenge_solution',
      version='0.1',
      description='Brief Description of your AI component',
      author='Your_team',
      author_email='Your_email',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=parse_requirements('requirements.txt')
     )