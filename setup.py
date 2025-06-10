from distutils.core import setup

setup(name = 'sr', 
      version = '1.0',
      description = 'Symbolic regression',
      author = 'Julien LIVET',
      author_email = 'julien.livet@free.fr',
      install_requires = ['sympy', 'numpy', 'scipy', 'deap', 'matplotlib', 'scikit-learn'],
      setup_requires = ['sympy', 'numpy', 'scipy', 'deap', 'matplotlib', 'scikit-learn'])
