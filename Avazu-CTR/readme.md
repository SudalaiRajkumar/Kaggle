This folder consists of the codes written for the [Avazu - Click Through Rate](https://www.kaggle.com/c/avazu-ctr-prediction) Kaggle competition. 

The approach followed here is Follow The Regularized Leader - Proximal (FTRL). It is an online learning algorithm and the algorithm can be read from [Google paper](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) and this [paper](http://people.csail.mit.edu/romer/papers/TISTRespPredAds.pdf)

Thanks to Tintgru for the [base code of the algorithm](https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory) implemented in python 

Also from this [paper](http://quinonero.net/Publications/predicting-clicks-facebook.pdf), it is shown that feature interactions improved the performance in the similar problems. This idea is added to the base code to get better result.
