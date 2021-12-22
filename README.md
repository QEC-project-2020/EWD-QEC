# Effective weight decoder

Contains source code for https://arxiv.org/abs/2112.01977

High accuracy decoding scheme based on minimal information about shortest chain weights.

Karl Hammar, Alexei Orekhov, Patrik Wallin Hybelius, Anna Katariina Wisakanto, Basudha Srivastava, Anton Frisk Kockum, Mats Granath

Contains source code for https://arxiv.org/abs/2112.06036, "The XYZ<sup>2</sup> hexagonal stabilizer code", Basudha Srivastava, Anton Frisk Kockum, Mats Granath

## Prerequisites 
- Python 3
- (optional) *venv* or other environment manager

## Getting started 
### Installation 
- Clone the repository:
```bash
git clone git@github.com:QEC-project-2020/EWD-QEC.git
```
- Install the requirements with venv:

```bash
python -m venv ewd
source ewd/bin/activate
pip install -r requirements.txt
```
## How to use the decoder

To run a simple test of the decoder, the following file is provided:
- run.py

This script will in turn run the desired decoder for the desired number of steps on the desired code. This is where syndroms are generated and data is saved.

The high level decoding code is found in ***decoders.py***, and the more fine-grained details of the MCMC simulations and code definitions can be found in the ***src*** directory.

The data is saved in *pandas* DataFrames, where the first row contains a copy of the parameter set used for the simulation, and the following rows contain the generated error chain and corresponding decoder predictions. An example on how to read the file and plot the results can be seen in:
- plot.py

### Use of MWPM
The MWPM decoder (and the eMWPM flavour) can be used either as a decoder or as a tool to find initial chains for other algorithms to use. The MWPM decoder is based on *blossom5* algorithm which is available at https://pub.ist.ac.at/~vnk/software.html (for research purposes only). Download and extract the tarball in the ***src*** directory to use it.
## Structure of this repository

File | Description
----- | -----
`├── data` | A directory that contains error correction simulations.
`├── plots` | A directory containing plots.
`├── src` | Source files utility code for decoders.
`·   ├── mcmc_alpha.py` | MCMC methods with alpha noise parametrization.
`·   ├── mcmc_biased.py` | MCMC methods with biased noise parametrization.
`·   ├── mcmc.py` | MCMC methods for depolarizing noise.
`·   ├── mwpm.py` | MWPM decoder and compability layer.
`·   ├── planar_model.py` | Implementation of the planar code.
`·   ├── rotated_surface_model.py` | Implementation of the rotated surface code.
`·   ├── toric_model.py` | Implementation of the toric code.
`·   ├── xzzx_model.py` | Implementation of the XZZX code.
`·   └── xyz2_model.py` | Implementation of the XYZ<sup>2</sup> code.
`├── decoders.py` | Implementation of the EWD decoder among others.
`├── generate_data.py` | Data generation script.
`├── LICENCE` | Licene for this project.
`├── plot.py` | Example file to plot error correction simulations.
`├── README.md` | About this project.
`├── requirements.txt` | Required packages to run this project. Other versions may work but not tested.
`└── run.py` | Example file to run a simple error correction simulation.
