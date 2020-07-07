# Analysis of Predictive Coding Models for Phonemic Representation Learning in Small Datasets

Neural network models using predictive coding are interesting from the viewpoint of computational modelling of human language acquisition, 
where the objective is to understand how linguistic units could be learned from speech without any labels. Even though several promising predictive coding -based 
learning algorithms have been proposed in the literature, it is currently unclear how well they generalise to different languages and training dataset sizes. 
In addition, despite that such models have shown to be effective phonemic feature learners, it is unclear whether minimisation of the predictive loss functions 
of these models also leads to optimal phoneme-like representations. The present study investigates the behaviour of two predictive coding models, Autoregressive 
Predictive Coding and Contrastive Predictive Coding, in a phoneme discrimination task (ABX task) for two languages with different dataset sizes. 
Our experiments show a strong correlation between the autoregressive loss and the phoneme discrimination scores with the two datasets. However, to our surprise, 
the CPC model shows rapid convergence already after one pass over the training data, and, on average, its representations outperform those of APC on both languages.

### Python module

This folder includes: 
* the APC and CPC models' implementation `python_module/models`. The implementatio of APC model is an adaptation for Keras of the
implementation by Chung et al. (https://github.com/iamyuanchung/Autoregressive-Predictive-Coding). For the CPC model, we adapted the implementation of the
contrastive loss of the Wav2Vec model (Schneider et al.) (https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec.py).


* Script to obtain the scatter plots and correlation coefficients `python_module/statistical_analysis.py`

### Data
This folder includes the data of the different experiments and the scatter plots.


JSON files:

* model\_language\_data.json: it contains the data points for each run in the form 
(epoch id, validation loss value, ABX across-speaker score, ABX within-speaker score)

* model\_language\_stats.json: it contains the statistical measures calcualted for the average performance 
and for each run. 

### Publication

"Analysis of Predictive Coding Models for Phonemic Representation Learning in Small Datasets". María Andrea Cruz Blandón, Okko Räsänen. 
Proceedings of the 37th International Conference on Machine Learning, PMLR 108, 2020.


 
