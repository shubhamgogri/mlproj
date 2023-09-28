from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_req(file_path:str)-> List[str]:
    '''
    Returns the list of reqiurements 
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # removing the \n from the requirements 
        requirements = [req.replace("\n", "") for req in requirements]
        # while executing the setup.py '-e .' should not be there in the list otherwise it should be present in the requirements file. 
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(

    name= 'mlproject',
    version='0.0.1',
    author='shubham', 
    author_email='gogrishubham85@gmail.com',
    packages=find_packages(),
    install_requirements = get_req('requirements.txt')
)