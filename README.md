# gr-rnn-sentence-classification 

## Description
This project is targeted for sentence classification by RNN structure. 

## Requirements
This project require following environments

|Environment|Version|
|:---------:|:-----:|
| Python | 3.6.x or later |
| TensorFlow | 1.3.x |
| numpy | 1.13.x |

## Dataset
This project use following dataset for train and test.  
[Movie Review data from Rotten Tomatoes](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

## Reference of Dataset
[Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.home.html)

## Usage

Learning Step:  
Use ```sentece_classification.py```. Once executed, learning is executed with the default hyperparameter and model data and vocabulary are generated under the "run" directory.  

Classification App:  
Use ```Classifier.py```. Once execute, classification is executed with the default hyperparameter and the classification result of the test sentence written in the program is output.  

If you want to print all hyper parameter and runtime parameter, please execute the following command.  
```
python sentence_classification.py --help
```  
or  
```
python Classifier.py --help
```  

## License
This software is released under the Apache License 2.0, see LICENSE.
