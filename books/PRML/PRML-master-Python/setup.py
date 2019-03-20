from setuptools import setup, find_packages


setup(
    name="prml",
    version="0.0.1",
    description="Collection of PRML algorithms",
    author="ctgk",
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
    packages=find_packages(exclude=["test", "test.*"]),
    test_suite="test"
)
