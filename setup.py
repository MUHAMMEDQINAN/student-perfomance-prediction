from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e.'


'''
SUMMARY:
get_requirements reads the content of a requirements file and returns a list of the requirements (dependencies) specified in it, excluding certain entries like -e .
This defines a function named get_requirements, which accepts a single argument file_path of type str. The function is expected to return a list of strings, as indicated by -> list[str].
An empty list requirements is created to store the lines (dependencies) read from the requirements.txt file.
with open(file_path) as file_obj statement opens the file at the path specified by file_path.
The readlines() method is called on file_obj, which reads all the lines of the file and returns them as a list of strings. Each string in the list represents a line from the file, including any newline characters (\n) at the end of each line.
checks if the string '-e .' (the value of HYPHEN_E_DOT) exists in the requirements list.
If it does, it removes the '-e .' entry from the list, as this is not typically a dependency you want to include in the list of requirements that you might install.
Finally, the function returns the requirements list, which now contains the dependencies (without the -e . entry, if present) as plain strings.
'''
def get_requirements(file_path:str)->List[str]:
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'student Performance Prediction',
    version='0.1',
    author='Muhammed Qinan',
    author_email='muhammedqinanpk@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)