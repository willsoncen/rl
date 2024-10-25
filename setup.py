from setuptools import setup, find_packages

setup(
    name="hierarchical-rl-traffic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "sumolib>=1.8.0",
        "traci>=1.8.0",
        "pyyaml>=5.4.1",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="分层强化学习交通控制系统",
    keywords="reinforcement-learning, traffic-control, SUMO",
    python_requires=">=3.8",
)

