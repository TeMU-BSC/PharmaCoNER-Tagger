# PharmaCoNER Tagger

## Introduction

PharmaCoNER Tagger is a Neural Named Entity Recognition program targeting domain adaptation, particularly in the case of Spanish medical texts. It is based on an already existing state-of-the-art neural NER, NeuroNER[^1], and we refer to its original article, documentation and source code for further information.

PharmaCoNER Tagger was developed within PlanTL[^2] in the Barcelona Supercomputing Center[^3].

Both Docker and Singularity images will be available.

### Features added to NeuroNER

NeuroNER is designed to have text as input. However, some domains, like the Spanish medical texts, present some challenges. Real-world medical data is scarce, so dealing with low-resource datasets is a concern. Also, words referring to pharmacological compounds and medical concepts can be morphologically complex. For these reasons, the following features were added to NeuroNER:
- Part-Of-Speech tags: Apart from the original tokens, as in NeuroNER, PharmaCoNER Tagger can optionally be input tags referring to the original tokens. The main use case when developing was being able to use Part-Of-Speech tags extracted by a domain-specific POS tagger[^4], but other tags like lemmas could be used as well.
- Gazetteer features: Non-neural NER for chemical compounds used features extracted from dictionaries. We incorporate this factor by adding an optional boolean (binary) feature that denotes whether each token belongs to the dictionary (provided by the user) or not.
- Affixes: Similarly to the gazetteer, the user can define a set of affixes (for instance, related to pharmacological compounds). PharmaCoNER Tagger will check tokens against these affixes and set a boolean variable accordingly.

Notice that these features added to the neural network could be used for different domains and therefore are not necessarily domain-specific. To leverage these additions in the case of the Spanish medical texts, some preprocessing steps were required. In our experiments, we applied our domain-specific POS tagger, build a gazetteer based on a pharmaceutical nomenclator maintained by the Spanish government and tried 8 different embeddings, both general and domain-specific. In the next sections, we will detail how to apply these steps.

### HPC

Even if NeuroNER supports running on a GPU, it is known not to leverage its capabilities[^5]. We installed PharmaCoNER Tagger in BSC's HPC clusters with a virtual environment. After having discarded the use of GPUs, we tried different numbers of cores. In the usage section, we will provide both Docker and Singularity images.

## Prerequisites

PharmaCoNER Tagger is based on NeuroNER, which relies on Tensorflow, Spacy and other Python packages. Provided that NeuroNER requirements are met, PharmaCoNER Tagger should be easy to run in most machines. In the installation section, we provide instructions for installing both the dependencies and the program itself.

### Directory structure

- src/: Apart from the source code of the program itself, in these directories there are some scripts and utilities, like an example parameters.ini (used for configuring PharmaCoNER Tagger, as we will see later) and the script for exporting trained models (prepare_pretrained_model.py). 
- data/: Datasets should be located in this directory.
 - data/PharmaCoNERCorpus: The corpus used in the PharmaCoNER task, based on the SPACCC corpus, is already present in this directory and in the right format for PharmaCoNER Tagger to ease the quickstart of this example.
 - data/word_vectors: Word embeddings should be stored in this directory.
 - data/gazetteers: Gazetteer dictionaries should be located in this directory.
 - data/affixes: Affix dictionaries should be located in this directory.
- pretrained_models: Exported models are saved in this directory by default.

## Installation

PharmaCoNER Tagger can be installed by downloading or cloning this repository and installing its dependencies. It is recommended to do so in a virtual environment, like venv:

    git clone https://github.com/TeMU-BSC/PharmaCoNER-Tagger.git
    python3 -m venv PharmaCoNER-Tagger
    source PharmaCoNER-Tagger/bin/activate
    python -m pip install -r PharmaCoNER-Tagger/requirements.txt
    python -m spacy download es

These steps should be enough for most of the machines, provided they have a Python distribution. We refer to these two NeuroNER guides for further information on installation:
 - Installing NeuroNER on Windows: <https://github.com/Franck-Dernoncourt/NeuroNER/blob/ca344ea893f29ec8478686a6d7d31f3bd3f78c1f/install_windows.md>
 - Installing NeuroNER on Ubuntu: <https://github.com/Franck-Dernoncourt/NeuroNER/blob/ca344ea893f29ec8478686a6d7d31f3bd3f78c1f/install_ubuntu.md>

Alternatively, just download the Docker and Singularity images we are providing and run them with Docker or Singularity, respectively.


## Usage and examples

For example, we will see how to use PharmaCoNER Tagger in the PharmaCoNER task[^7], which consists of pharmacological substance, compound, and protein named entity recognition. It is based on the SPACCC corpus[^8]

### Data

Datasets should be located in the data/ directory. If they are formatted in BRAT, a sub-directory for each subset should be created with the following names:
 - train/: The training set is required if PharmaCoNER Tagger is running in training mode.
 - valid/: Optional directory with the validation/development set.
 - test/: The test set is required if PharmaCoNER Tagger is running inraining mode.
 - deploy/: The deployment/production set is required if PharmaCoNER Tagger is running in deployment mode.
As mandated by the BRAT format, each text in these subsets should have a plain text file with .txt extension containing the text, and another file, with the same name but .ann extension, with the corresponding annotations. For more information, see [^9].

Alternatively, datasets can be provided in CoNLL, with train.txt, valid.txt, test.txt and deploy.txt files.

In our example, the PharmaCoNER task corpus is in BRAT format and it is located in data/PharmaCoNERCorpus. There are 4 kinds of entities: PROTEINAS, NORMALIZABLES, NO-NORMALIZABLES and UNCLEAR. Notice that the dataset is imbalanced (ie. there are many more NORMALIZABLES and PROTEINAS than the other two ones), which is challenging. Techniques for dealing with imbalanced datasets such as data augmentation and oversampling could be tried.

Setting the dataset is as easy as modifying the following parameter in src/parameters.ini to the desired path:
    [dataset]
    dataset_text_folder = ../data/PharmaCoNERCorpus
Notice that the parameters.ini provided by this repository is already set to the PharmaCoNER task corpus.

### Word embeddings

NeuroNER can load pre-trained word embeddings instead of randomly initializing them. In the case of our task, it is particularly clever to leverage learning transfer, since it is a low-resourced dataset. For this example, we are going to use the GloVe embeddings from SBWC[^10], trained by Universidad de Chile. However, we encourage the user to try other embeddings. In the same repository from Universidad de Chile, other embeddings can be found, and the Fasttext ones are probably going to perform better. Besides, we recommend to try out Spanish domain-specific embeddings for medical texts[^11], some of which performed better in our experiments.

The word embeddings must be downloaded by the user and stored in data/word_vectors/. If the word_vectors/ subdirectory does not exist, it must be created. As we said, in our example, we are going to use the GloVe embeddings from SBWC, and the file can be downloaded from <http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz>. Once downloaded, it should be uncompressed and moved to data/word_vectors/:
    wget http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz
    gunzip glove-sbwc.i25.vec.gz
    mkdir data/word_vectors
    mv glove-sbwc.i25.vec word_vectors/

Notice that this repository does not include the aforementioned embeddings.

In parameters.ini, the user can specify the desired word embeddings to use. The GloVe embeddings from SBWC are defined by default.

    token_pretrained_embedding_filepath = ../data/word_vectors/glove-sbwc.i25.vec
    token_embedding_dimension = 300
    token_lstm_hidden_state_dimension = 300
 

Both token_embedding_dimension and token_lstm_hidden_state_dimension should be set to the token embedding dimension, which is specified by the developer of the pre-trained embeddings. Instead, if the user wants to randomly initialize the token embeddings, the field must be left as empty.
    token_pretrained_embedding_filepath = 

### Basic usage

Once PharmaCoNER Tagger has been installed and the pre-trained GloVe embeddings have been downloaded and moved to the right directory, for running PharmaCoNER with the task corpus execute:
    cd src/ & python main.py
    
Assuming the virtual environment is activated and that we are in the root of this repository.

The logs will show the performance (with metrics such as accuracy and FB1 score) of the system for each epoch, as well as other execution details. The training is set to early stop after 10 epochs of no improving the results in the validation set, with a maximum of 100 epochs, but this can be easily configured in the parameters.ini file as well.

With this configuration, we find the best results in epoch 35, with the training being stopped in epoch 45, with the following validation evaluation:

    processed 98075 tokens with 1915 phrases; found: 1802 phrases; correct: 1554.
    accuracy:  99.34%; precision:  86.24%; recall:  81.15%; FB1:  83.62
    NORMALIZABLES: precision:  88.60%; recall:  82.81%; FB1:  85.61  1044
    NO_NORMALIZABLES: precision:   0.00%; recall:   0.00%; FB1:   0.00  1
    PROTEINAS: precision:  82.94%; recall:  81.03%; FB1:  81.97  721
    UNCLEAR: precision:  86.11%; recall:  70.45%; FB1:  77.50  36

For further and more advanced configuration details not directly related to PharmaCoNER Tagger modifications, we refer to the comments inserted in the parameters.ini and NeuroNER's documentation.

### Preprocessing

For the basic usage, apart from the need of providing the dataset in either BRAT or CoNLL, there is no need for any preprocessing, because NeuroNER performs the tokenization step itself (using Spacy's tokenizer by default).

### Model exporting and deployment mode

In order to export a model, edit the last lines of src/prepare_trained_model.py:
    output_folder_name = 'my_output_folder' # name of the output
    epoch_number = 35 # epoc
    model_name = 'my_model_name' # name 

With regard to output_folder_name, notice that PharmaCoNER Tagger will create a new directory for each training execution in the output folder (by default, in output/). The name of these training directories will be the experiment name ('test', by default, but can be modified in the parameters.ini file, in experiment_name field) plus a timestamp. Identify the desired directory and set the field output_folder_name in src/prepare_trained_model.py accordingly.

Then, run (assuming the virtual environment is activated and the user is in src/):
    python prepare_trained_model.py

The script will create a folder with the exported model in the directory trained_models/.

To run the exported model, edit the beginning of the trained_models/<model_name>/parameters.ini file:
    train_model = True # Set to true to keep training the model from the exported checkpoint. Set to false for deployment mode.
    use_pretrained_model = True # In order to use a pretrained model, it should always be set to true 
    pretrained_model_folder = ../trained_models/<model_name> # pre-trained model path

Then, execute (again, assuming the virtual environment is activated and the user is in src/):
    python main.py --parameters_filepath trained_models/<model_name>/parameters.ini

### Adding domain features

#### Options
In the parameters.ini file, there are options for using the aforementioned domain features, although they are deactivated by default. In this section, we will detail how to use them:
- Part-Of-Speech: The parameters ‘use_pos’ (boolean) ‘freeze_pos’ (boolean) have been added. If ‘use_pos’ is set to true, then NeuroNER parses POS tags too, in dataset.py. If the dataset is written in CONLL format, dataset.py just takes the POS tags from the corresponding column. If the dataset is written in BRAT format, in brat_to_conll.py, apart from parsing the ‘.ann’ files, NeuroNER parses ‘.ann2’ files, too, which should contain the POS tags. If the tokenization and splitting of the POS tags do not come from Spacy, then the parameter ‘tokenizer’ should be set to ‘pos’. Otherwise, the tokenization and splitting could be incompatible. Once the CONLL files have been created, dataset.py can parse the POS tags. Once parsed, also in dataset.py, the POS tags are encoded with a one-hot encoding vector, similarly to how the original NeuroNER transformed the labels. Indices to this vector are created, in the same manner as was done in the original codebase for the rest of the features. The neural network receives the one-hot vector, in entity_lstm.py. If ‘use_pos’ is set to true, then the vector of POS tags is concatenated with the vector of token embeddings, or the vector of token and character embeddings, depending on the parameters. If ‘freeze_pos’ is set to True, then the weights of the POS tags vector are not trained.
- Gazetteer features: The parameters ‘use_gaz’, ‘freeze_gaz’ and ‘gaz_filepath’ have been added. The gazetteer must be a plain text file with one word at each line, and its path is specified with the parameter ‘gaz_filepath’. The parameters ‘use_gaz’ and ‘freeze_gaz’ have the same meaning as before. In dataset.py, if ‘use_gaz’ is set to true, then the gazetteer file is parsed. For each token, if the token is present in the gazetteer, then the gazetteer feature for the token will be set to 1, otherwise, it will be set to 0. Indices for the vector of gazetteer features are created similarly as before. In entitiy_lstm.py, this variable is similarly concatenated to the other vectors as before.
- Affixes: Affixes work similarly as the aforementioned gazetteer features, with parameters 'aff_filepath', 'use_aff', and 'freeze_aff'. However, instead of just checking the existence of the whole token in the dictionary, in the case of affixes, PharmaCoNER uses regular expressions to check if words start or end by the corresponding affix (depending on whether the affix is a prefix or a suffix). The affix dictionary contains both the English and Spanish forms, and the regular expression checks against both cases. Specifically, the dictionary should be a .tsv file. Later on, we describe the format (there is a sample file provided in this repository).  

#### Resources

- POS: As a domain-specific POS tagger, we recommend the SPACCC POS Tagger [^4].
- Gazetteer: In data/gazetteers/gazetteer.txt we are including a sample gazetteer built with a pharmaceutical nomenclator maintained by the Spanish government. 
- Affixes: In data/affixes/affixes.tsv we are including a sample affix dictionary with common prefixes and suffixes related to pharmaceutical compounds.

#### Preprocessing needed for using domain features

- POS: POS tags should be in BRAT format. Each annotation file should be in the same directory as its corresponding text (and with the same filename). To avoid conflicts with the named entities annotations, the file extension should be changed to .ann2 (instead of ann).
- Gazetteer: Just move the gazetteer dictionary to data/gazetteers (and modify the path in parameters.ini accordingly), with the same format as the sample. 
- Affixes: Just move the affixes dictionary to data/gazetteers (and modify the path in parameters.ini accordingly), with the same format as the sample. 

## Demo

An online demo, pre-trained with the PharmaCoNER corpus, will soon be available at TeMU's webpage.

## Contact

For further information on the experiments and downloading the citation, see our original article in [^6]. For any doubt or comment, please contact:
Jordi Armengol <jordi.armengol@bsc.es>

## License

PharmaCoNER Tagger is based on NeuroNER (MIT Copyright (c) 2019 Franck Dernoncourt, Jenny Lee, Tom Pollard), which has a MIT license (see [^1]. Notice that PharmaCoNER Tagger is not affiliated, associated, authorized, endorsed by, or in any way officially connected with NeuroNER or their authors. With regard to the license of the dataset provided in this repository, SPACCC, see [^8]. PharmaCoNER itself has a MIT license as well.

## References

 [^1]: NeuroNER (MIT Copyright (c) 2019 Franck Dernoncourt, Jenny Lee, Tom Pollard): <https://github.com/Franck-Dernoncourt/NeuroNER>.
 
 [^2]: PlanTL: Plan de impulso de Tecnologas del Lenguaje: <https://www.plantl.gob.es/sanidad/Paginas/sanidad.aspx>
 
 [^3]: Barcelona Supercomputing Center - Centro Nacional de Computacin (BSC - CNS): <https://www.bsc.es>.
 
 [^4]: SPACCC_POS-TAGGER: Spanish Clinical Case Corpus Part-of-Speech Tagger: <https://github.com/PlanTL-SANIDAD/SPACCC_POS-TAGGER>.
 
 [^5]: It is a known issue. See <https://github.com/Franck-Dernoncourt/NeuroNER/issues/3>.
  
 [^6]: PharmaCoNER Tagger: a deep learning-based tool for automatically finding chemicals and drugs in Spanish medical texts: <https://genominfo.org/journal/view.php?number=557>.
 
 [^7]: PharmaCoNER task: <http://temu.bsc.es/PharmaCoNER>.
 
 [^8]: SPACCC: Spanish Clinical Case Corpus: <https://github.com/PlanTL-SANIDAD/SPACCC>.
 
 [^9]: BRAT format: <https://brat.nlplab.org/standoff.html>.
 
 [^10]: Universidad de Chile: Spanish Word Embeddings: <https://github.com/dccuchile/spanish-word-embeddings>.
 
 [^11]: [PlanTL/medicine/word embeddings] Word embeddings generated from Spanish corpora: <https://github.com/PlanTL-SANIDAD/Embeddings>.
