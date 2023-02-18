import wheel
import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
  
setuptools.setup(
    name = 'pyratlib',         
    packages = ['pyratlib'],   
    version = '0.7.7',      
    license='MIT',       
    description = 'PyRat is a user friendly library in python to analyze data from the DeepLabCut. Developed to help researchers unfamiliar with programming can perform animal behavior analysis more simpler.',   # Give a short description about your library
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/pyratlib/pyrat',  
    download_url = 'https://github.com/pyratlib/pyrat',    
    keywords = ['Data analysis', 'Animal Behavior', 'Electrophysiology', 'Tracking', 'DeepLabCut'],   
    install_requires=[           
            'numpy',
            'pandas',
            'neo',
            'scikit-learn',
            'wheel'
        ],
    classifiers=[
      'Development Status :: 4 - Beta',      
      'Intended Audience :: Developers',      
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: MIT License',   
      'Programming Language :: Python :: 3',      
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
    ],   
)