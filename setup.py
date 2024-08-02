from setuptools import find_packages, setup

# with open("src/readme.md","r") as f:
#     long_description= f.read()

setup(
    name="Evaluation_Of_Safety_Oriented_Metrics_For_Object_Detectors",
    version="0.0.1",
    description="We argue that object detectors in the safety "
                "critical domain should prioritize detection of objects that are most likely to interfere "
                "with the actions of the actor, especially when they can impact task safety and reliability.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="MuhammadIbtihajNaeem",
    author_email="pro.ibtihajnaeem@gmail.com",
    license="",
    classifiers=[
        "License :: ",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'nuscenes-devkit==1.1.10',
    ],
    python_requires=">=3.7.9"
)
