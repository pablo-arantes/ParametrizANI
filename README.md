# ParametrizANI
Dihedral Parametrization in the Cloud with TorchANI

<img width="1584" alt="image" src="https://user-images.githubusercontent.com/35934150/191140934-d11276fc-2d51-48e5-bf65-87aee84e03a4.png">


## Hi there!

This is a repository where you can find a Jupyter notebook scripts to set up a protocol for parametrization of small molecules dihedrals for GAFF and OpenFF force fields using TorchANI, a PyTorch-based program for training/inference of ANI (ANAKIN-ME) deep learning models to obtain potential energy surfaces and other physical properties of molecular systems.  TorchANI is open-source and freely available on GitHub: https://github.com/aiqm/torchani.

The main goal of this work is to demonstrate how to harness the power of cloud-computing to parametrize compounds in a cheap and yet feasible fashion.

**ParametrizANI** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI.ipynb)  - `Dihedral Parametrization with TorchANI as a reference and download the topology in AMBER, GROMACS and OpenMM format.`

## Bugs
- If you encounter any bugs, please report the issue to https://github.com/pablo-arantes/ParametrizANI/issues

## Acknowledgments
- ParametrizANI by **Souvik Sinha** ([@sou_svk](https://twitter.com/sou_svk)) and **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes))
- We would like to thank the OpenMM team for developing an excellent and open source engine. 
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
