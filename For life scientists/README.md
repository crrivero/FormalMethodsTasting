# For life scientists

Every notebook here is self-contained and meant to be opened in Google Colab; see the [repository README](../README.md) for how to run one in class. Each entry below says what the student is asked to write and which part of the Z3 primer the notebook assumes.

All three notebooks in this folder are currently demonstrations: the student reads and runs them to see what an SMT solver can do, rather than filling anything in.

You can find a number of Jupyter notebooks for:

- Systems biology:
  + Find a clique in a protein-protein interaction network [link](PPI-CLIQUE.ipynb). <details><summary>Click to expand description</summary>Given a network, a clique is a subset of nodes in the network that are all adjacent to oneanother. The objective of this notebook is to use Z3 to find a clique in a Protein-Protein Interaction Network. Each protein becomes a boolean variable, a constraint rules out selecting any pair that does not interact, and a second constraint forces at least k proteins to be chosen; the network and the clique found are both drawn. The whole procedure is then repeated on a second network built around p53 and its interactors. <br><br> <b>Student task:</b> none — this notebook is read and run rather than filled in. <br> <b>Z3 primer:</b> booleans.</details>

- Molecular biology:
  + Recover the DNA behind a protein [link](REVERSE-TRANSCRIPTION-TRANSLATION.ipynb). <details><summary>Click to expand description</summary> Going from a DNA template strand to a protein is mechanical: transcribe to mRNA, then read the mRNA three bases at a time. Going the other way is not, because most amino acids are coded for by several different codons, so a protein does not determine its own mRNA. The notebook writes the forward direction as ordinary Python first, then describes to Z3 what a valid mRNA strand for a given protein looks like and asks the solver to enumerate the possibilities — including on a full codon table, where there are a great many of them. <br><br> <b>Student task:</b> none — this notebook is read and run rather than filled in. <br> <b>Z3 primer:</b> integers.</details>

- Population genetics:
  + Pair off a population by who can mate with whom [link](MATING-MATCHING.ipynb). <details><summary>Click to expand description</summary> Not every individual in a population can breed with every other one; genetics, blood type and other factors rule some pairings out. Given which pairings are possible, can everyone be paired off at once? Each possible pairing becomes a boolean variable, and the constraints say what it means for a set of pairings to be a valid matching: nobody is paired twice, and everyone is paired. The population and the matching found are both drawn. <br><br> <b>Student task:</b> none — this notebook is read and run rather than filled in. <br> <b>Z3 primer:</b> booleans.</details>
