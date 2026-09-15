# A Taste of Formal Methods in the Field
We aim to improve formal methods exposure by developing self-contained Jupyter notebooks (using satisfiability modulo theories solvers accessed via Python) that focus on small-scale problems. These small-scale problems give a taste of formal methods to different STEM communities, namely: three institutions (a research institution, a liberal-arts college, and a community college), several computing and engineering majors (computer science, bioinformatics, computer engineering, chemical engineering, and more), and several topics and levels (introductory programming, engineering design, digital system design, databases, artificial intelligence, and more).

You can find notebooks for the following disciplines. Each folder has its own README describing every notebook it contains, what the student is asked to write, and which part of the Z3 primer the notebook assumes.

+ [For computer scientists](For%20computer%20scientists/) — computer organization, databases, algorithm analysis, theory, artificial intelligence, introductory programming
+ [For engineers](For%20engineers/) — computer, mechanical, electrical, and optical engineering
+ [For physicists](For%20physicists/) — mechanics, electricity, optics
+ [For chemical engineers](For%20chemical%20engineers/) — reaction stoichiometry, molecular structure
+ [For life scientists](For%20life%20scientists/) — molecular biology, systems biology, population genetics
+ [For mathematicians](For%20mathematicians/) — number theory, graph theory

A few notebooks suit more than one audience and appear in more than one folder; those copies are identical, and each folder's README says so.

Every notebook is self-contained. Its first cell installs Z3 and our helper library, a short primer introduces the Z3 constructs that notebook needs, and the rest is a single activity in the student's own subject. Most notebooks ask the student to write one or a few constraints; some are demonstrations that are read and run rather than filled in.

The `core/` folder holds the pieces the notebooks are built from: `tofmcore.py`, the helper library every notebook imports, and the generator that assembles a notebook from an activity core plus the shared intro, primer, and outro.

## Running notebooks in your class

To get started, choose a notebook of your liking
and have your students copy it to their Google drive and have them go through it using Google Colab. The notebooks are intended to be run on Google Colab, and we recommend using it to avoid any compatibility issues that may arise from local installations of Jupyter.

# Supported by NSF
Award #2421977 (https://www.nsf.gov/awardsearch/showAward?AWD_ID=2421977)

Award #2421978 (https://www.nsf.gov/awardsearch/showAward?AWD_ID=2421978)
