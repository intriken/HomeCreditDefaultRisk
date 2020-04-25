Python Libraries needed:

pip install kaggle
pip install numpy
pip install lightgbm
pip install sklearn
pip install matplotlib
pip install seaborn


Jar to create PPML file from model file
using build 1.2.14 from https://github.com/jpmml/jpmml-lightgbm

java -jar jpmml-lightgbm-executable-1.2.14.jar --lgbm-input ../model/gbm_classifier.txt --pmml-output ../model/gbm_classifier.pmml
