### An adaboost implementation in Python.

This algorithm was developed in order to help me understand 
how the original algorithm by Robert Shapire works.
<br>
**This his not a trusty implementation.**
For professional or academic purposes
please use a good adaboost implementation like the one provided in the scikit-learn library.

For some math reference you can check:
http://rob.schapire.net/papers/explaining-adaboost.pdf

**Note**: I did not follow the exactly steps of the original algorithm.
I dit not use the sign function but instead I perform a min/max normalization of all the values to obtain a probability. 
From this probability and using a defined threshold, I then predict if the true value was zero 0 or 1. 
Also, this algorithm his based on my **personal interpretation**(wich may not
be correct) of the original algorithm.

