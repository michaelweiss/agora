from setuptools import setup

setup(name='agora',
      version='0.1.0',
      author='Michael Weiss',
      packages=['agora'],
      description='A framework for building agent workflows.',
      license='MIT',
      install_requires=[
            'anthropic',
            'openai',
            'streamlit',
      ],
)