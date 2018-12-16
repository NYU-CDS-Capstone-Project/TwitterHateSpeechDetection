These files are intended to in conjunction with the files found here: https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts.

This folder includes

(1) tok2id_mod.py, which modifies tok2id.py such that it loads a pre-existing vocabulary file rather than creating a new one.

    This is needed when the classifier is based on a different dataset than the fine-tuned language model.
    
(2) scripts to train and evaluate fine-tuned language models and classifiers.

