# uncertain_position_limits
Numerical code to calculate constraints on the emission from a source whose position is uncertain and given by PDF

toymc/toy_paper.cpp - master script that can do all the calculations with toy MC 

tilepy/simulate.ipynb - notebook which prepares the input for calculation. It takes the standard IRFs, probability distribution of the GW, and list of pointings and prepares the averaged IRFs 
tilepy/loadandrun.ipynb - takes the input of simulate.ipynb to compare the three methods on a single realization. For the frequentist approach it needs the library of the MC realization done with the freq/freq_single.py script
gen_many/gen_many_ne.py - slightly optimized (for parallel computing) version of the algorithms to test it on many realizations (for comparisons)
