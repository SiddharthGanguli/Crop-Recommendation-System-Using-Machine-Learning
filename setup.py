from setuptools import setup,find_packages

setup(
    name="crop_recommendation_system",
    version='1.1.1',
    author="Siddhartha Ganguli",
    author_email="siddharthaganguli0093@gmail.com",
    description="Crop Prediction Machine Learning System",
    license="MIT",

    package_dir={'':'src'},
    packages=find_packages(where='src'),
    
    include_package_data=True,
    python_requires=">=3.8"
)

