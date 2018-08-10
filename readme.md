My second neural network experiment (first was FIR filter). 
DFT output is just a linear combination of inputs, so it should be 
implementable by a single layer with no activation function.

Animation of weights being trained:

![Neural network weights heatmap](https://i.imgur.com/5SjyBsw.gif)

Red are positive, blue are negative.
The black squares are unused, and could be pruned out (if I knew how to do that).

Even with pruning it would be less efficient than an FFT, so if 
the FFT output is useful, probably best to provide it as separate inputs?