# Reproducing Axelrod's Evolution of Strategies in the Iterated Prisoner's Dilemma
A development project focused on reproducing results found from Robert Axelrod's unreleased third tournament with genetic algorithms, found [here](https://www.cse.unr.edu/~simingl/papers/DSA/Evolving%20New%20Strategies%20Iterated%20Prisoner%20Dilemma.pdf). The open source Axelrod-Python library created and maintained by the wider IPD community was crucial to the success of this replication, visit them [here](https://axelrod.readthedocs.io/en/stable/index.html) for more documentation.

### Getting Started
This project was implemented on a Debian 13 machine in Python 3.13.5 and the latest HTML version. Dependencies include Numpy 2.2.4, Matplotlib 3.10.1, and Axelrod 4.14.0 modules. Dependencies can be installed with the following commands:
```
apt install python3-numpy
apt install python3-matplotlib
pip install axelrod
```

To replicate the project, clone the github repository, ensure the mentioned dependencies are installed. The NonEvolvingAB file is used for running both the non-evolving experiments, while the CoEvolution file is for the co-evolving experiment with a standard GA. The CoEvolutionElitism file runs both experiments involving the elitist GA and parameter modification.

To run experiment 1 with TFT (NonEvolving_A):
```
python3 NonEvolvingAB.py
```
Experiment 2 with TFT absent (NonEvolving_B):<br/>
In lines 355-356 of the file, comment out the the Tit for Tat strategy and uncomment the FELD strategy before running the same file as normal.
```
# axl.TitForTat()
axl.FirstByFeld()
```
Experiment 3 with co-evolving standarad GA:
```
python3 CoEvolution.py
```
Experiment 4 with co-evolving Elitist GA:
```
python3 CoEvolutionElitism.py
```
Experiment 5 with co-evolving Elitist GA + Parameter Modification:<br/>
In lines 17-24, change the following constants before running:
```
POPULATION_SIZE = 40   # Default population is 20
GENERATIONS = 100      # Default generations is 50
ELITISM_COUNT = 4      # Default set elites is 2
```

## Interactive Interface Demo
An interactive web UI has been created, which displays statistics and results from experiments 1, 2, and 4, along with a graph gallery which visualizes average mean score results for all experiments. A game also allows the user to load some of the best chromosomes found in this work or play against a custom one, either by manually pasting or uploading a JSON/TXT file containing a 70 bit chromosome. This webpage can be self hosted and does not have any external dependencies, as all images hae been converted to base64 and embedded directly into HTML. Below is a demo video of the interface.<br/>
https://github.com/user-attachments/assets/3ceaefe1-da48-4615-873c-7383c5873823


