.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

====
SaLS
====


    Method for the semantic segmentation of video sequences.


Over the last decade, the integration of robots into various applications has seen significant advancements fueled by Machine Learning (ML) algorithms, particularly in autonomous and independent operations. While robots have become increasingly proficient in various tasks, object instance recognition, a fundamental component of real-world robotic interactions, has witnessed remarkable improvements in accuracy and robustness. Nevertheless, most existing approaches heavily rely on prior information, limiting their adaptability in unfamiliar environments.

To address this constraint, this thesis introduces the Segment and Learn Semantics (SaLS) framework, which combines video object segmentation with Continual Learnin (CL) methods to enable semantic understanding in robotic applications. The research focuses on the potential application of SaLS in mobile robotics, with specific emphasis on the TORO robot developed at the Deutsches Zentrum für Luft- und Raumfahrt (DLR). Evaluation of the proposed method is conducted using a diverse dataset comprising various terrains and objects encountered by the TORO robot during its walking sessions.

The results demonstrate the effectiveness of SaLS in classifying both known and previously unseen objects, achieving an average accuracy of 78.86% and 70.78% in the CL experiments. When running the whole method in the image sequences collected with TORO, the accuracy scores were of 75.54% and 84.75%, for known and unknown objects respectively. Notably, SaLS exhibited resilience against catastrophic forgetting, with only minor accuracy decreases observed in specific cases. Computational resource usage was also explored, indicating that the method is feasible for practical mobile robotic systems, with GPU memory usage being a potential limiting factor.

In conclusion, the SaLS framework represents a significant step forward in enabling robots to autonomously understand and interact with their surroundings. This research contributes to the ongoing development of robotic systems that can operate effectively in unstructured environments, paving the way for more versatile and capable autonomous robots.

.. installation-instructions:

Installation
============

Here the installation instructions are described (it is recommended to install Mamba lib solver). Make sure to clone the main repository, init and update its modules.

Firstly, create a Python 3.8 environment::
    
    conda create -n lsuo python=3.8
    conda activate lsuo

Make sure to have the correct version of cuda and install pytorch and torchvision in https://pytorch.org/get-started/locally.

Then, install the requirements with conda::
    
    conda env update --file environment.yml

It can also be done with pip::
    
    pip install -e .

Install the requirements of SAM-Track::
    
    cd Segment-and-Track-Anything
    bash script/install.sh

Finally, download models automatically with::
    
    cd ..
    bash script/download_models.sh

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
