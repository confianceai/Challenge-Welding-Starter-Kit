from distutils.core import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename) as f:
        required = f.read().splitlines()
    return required



setup(name='challenge-welding',
      version='0.1',
      description='Set of tools functions to help challenge Welding users to interacts with challenge datasets',
      author='IRT SystemX',
      author_email='challenge.confiance@irt-systemx.fr',
      url='https://www.irt-systemx.fr/en/',
      packages=["challenge_welding"],
      install_requires=parse_requirements('requirements.txt')
     )