from distutils.core import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename) as f:
        required = f.read().splitlines()
    return required



setup(name='challenge_solution',
      version='0.1',
      description='An example of package containing a template solution for the challenge Weld quality detection',
      author='IRT SystemX',
      author_email='challenge.confiance@irt-systemx.fr',
      url='https://www.irt-systemx.fr/en/',
      packages=["challenge_solution"],
      license="TODO",
      install_requires=parse_requirements('requirements.txt')
     )