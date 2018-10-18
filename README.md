### An adaboost implementation in Python.

This algorithm was developed in order to help me understand 
how the original algorithm by Robert Shapire works.
<br>
**This his not a trusty implementation.**
For professional or academic porpouses
please use a good implementation like the one provided in the scikit-learn library.

For some math reference you can check:
http://rob.schapire.net/papers/explaining-adaboost.pdf

**Note**: I did not follow the exactly steps of the original algorithm.
I dit not use the sign function and instead I did a min/max normalization of all the values 
to give a predicted probability.From this probability I defined a threshold in order
to predict 0 or 1. Also this algorithm was based on my **personal interpretation**(wich might not
be correct) of the original algorithm.

