from setuptools import setup

setup(
    name="grail_unlearn",
    version="0.1.0",
    description="GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in LLMs",
    python_requires=">=3.9",
    packages=[
        "llm_unlearn",
        "llm_unlearn.methods",
        "llm_unlearn.utils",
        "llm_localization",
        "llm_evaluation",
    ],
)
