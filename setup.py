from setuptools import setup

setup(name = "TutorML",
    version = "0.1",
    description = "A machine learning library for tutorial",
    author = "Jianning Pan",
    author_email = "panjn@mail2.sysu.edu.cn",
    packages=["TutorML","TutorML.mixture","TutorML.demo.variational_bayes",
    "TutorML.utils"],
    install_requires = ['numpy','scipy','sklearn'],
)