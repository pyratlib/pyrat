import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
  
setuptools.setup(
    name = 'pyratlib',         # How you named your package folder (MyLib)
    packages = ['pyratlib'],   # Chose the same as "name"
    version = '0.1.5',      # Start with a small number and increase it with every change you make
    license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description = 'PyRat is a user friendly library in python to analyze data from the DeepLabCut. Developed to help researchers unfamiliar with programming can perform animal behavior analysis more simpler.',   # Give a short description about your library
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/pyratlib/pyrat',   # Provide either the link to your github or to your website
    download_url = 'https://github.com/pyratlib/pyrat',    # I explain this later on
    keywords = ['Data analysis', 'Animal Behavior', 'Electrophysiology'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
            'numpy',
            'pandas',
        ],
    classifiers=[
      'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
      'Intended Audience :: Developers',      # Define that your audience are developers
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: MIT License',   # Again, pick a license
      'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],   
)