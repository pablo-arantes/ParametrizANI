# ParametrizANI: Fast, Accurate and Free Parametrization for Small Molecules

![alt text](https://github.com/pablo-arantes/ParametrizANI/blob/main/TOC_graphic.png)

## Hi there!

Welcome to ParametrizANI, an innovative and free tool designed to address the growing demand for accurate parametrization of small molecules in molecular studies. Our goal is to democratize research by providing a research-friendly environment that is free from resource constraints, enabling teams of all sizes to perform dihedral parametrization with DFT-level accuracy.

**Key Features and Benefits:**

**• Robust Backbone:** ParametrizANI leverages TorchANI, a robust PyTorch-based deep learning program, as its benchmark to ensure precision in parametrization tasks. TorchANI is crucial for training and inference of ANI (ANAKIN-ME) deep learning models, which are fundamental for predicting potential energy surfaces and other molecular system attributes.

**• Accuracy and Efficiency:** ParametrizANI establishes detailed protocols for dihedral parametrization using both GAFF and OpenFF force fields. By integrating TorchANI's predictive power, ParametrizANI offers a streamlined and accurate approach to parametrization, especially for small molecules. TorchANI's neural network models predict molecular energies and properties with high accuracy and efficiency, significantly reducing computation time compared to traditional Quantum Mechanical (QM) methods.

**• Cloud-Based Accessibility:** The tool harnesses the power of Google Colaboratory (Colab), a hosted Jupyter Notebook service that provides free access to computing resources. This makes ParametrizANI a feasible, cost-effective, and accessible approach to compound parametrization, particularly beneficial for investigators worldwide, including those with limited resources. Our notebooks are designed to run efficiently on CPU cores, requiring no heavy parallel processing.

**• Comprehensive Workflow:** ParametrizANI provides comprehensive workflows implemented in Google Colab notebooks, exemplifying a complete pipeline for dihedral parametrization from SMILES strings generation to force field parameter optimization. These workflows enable researchers to efficiently perform accurate and reliable dihedral parametrization.

**• Versatile and Customizable:** The notebooks are designed for ease of use, following the Jupyter Notebook structure, with an initial configuration step taking less than 5 minutes. Users can select between GAFF and OpenFF force fields, choose charge calculation methods (AM1-BCC or RESP), and even upload their own reference energy profiles calculated using external software (e.g., Gaussian, GAMESS). This flexibility allows for customization to specific research requirements and professional use.

**• Broad Applicability:** ParametrizANI is not only suited for advanced molecular dynamics research and computational drug discovery but also serves as an excellent tool for educational purposes. It allows students to independently run the entire parametrization process without local software compilation or extensive coding experience, with embedded visualization at each step.


**ParametrizANI** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI.ipynb)  - `Dihedral Parametrization with TorchANI as a reference and download the topology in AMBER, GROMACS and OpenMM format.`

**TorchANI_2D** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/TorchANI_2D.ipynb) - `Two Dihedral scan with TorchANI and 3D plot of the map.`

**Psi4+TorchANI** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/Psi4%2BTorchANI.ipynb) -`Dihedral scan with Psi4 and structural optimization of each conformer with TorchANI.`

## Bugs
- If you encounter any bugs, please report the issue to https://github.com/pablo-arantes/ParametrizANI/issues

## Acknowledgments
- ParametrizANI by **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Souvik Sinha** and **Giulia Palermo**
- We would like to thank the OpenMM team for developing an excellent and open source engine.
- We would like to thank the [Psi4](https://psicode.org/) team for developing an excellent and open source suite of ab initio quantum chemistry.
- We would like to thank the [Roitberg](https://roitberg.chem.ufl.edu/) team for developing the fantastic [TorchANI](https://github.com/aiqm/torchani).
- We would like to thank the [Xavier Barril](http://www.ub.edu/bl/) team for their protocol on dihedrals parametrization and for the genetic algorithm script.
- We would like to thank [iwatobipen](https://twitter.com/iwatobipen) for his fantastic [blog](https://iwatobipen.wordpress.com/) on chemoinformatics.
- Also, credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin.
- Finally, we would like to thank [Making it rain](https://github.com/pablo-arantes/making-it-rain) team, **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Marcelo D. Polêto** ([@mdpoleto](https://twitter.com/mdpoleto)), **Conrado Pedebos** ([@ConradoPedebos](https://twitter.com/ConradoPedebos)) and **Rodrigo Ligabue-Braun** ([@ligabue_braun](https://twitter.com/ligabue_braun)), for their amazing work.

## How should I reference this work?
- For **TorchANI**, please cite: <br />
  Gao et al. "TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials." <br />
  Journal of Chemical Information and Modeling (2020) doi: [10.1021/acs.jcim.0c00451](https://doi.org/10.1021/acs.jcim.0c00451)
- For **OpenMM**, please also cite: <br />
  Eastman et al. "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics." <br />
  PLOS Computational Biology (2017) doi: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)
- For **Molecular Dynamics Notebook**, please also cite: <br />
  Arantes et al. "Making it rain: cloud-based molecular simulations for everyone." <br />
  Journal of Chemical Information and Modeling (2021) doi: [10.1021/acs.jcim.1c00998](https://doi.org/10.1021/acs.jcim.1c00998)
