I have left the data files out to reduce the size of the file to run any of the scripts please download the files from kaggle and put them in the input directory


Python Libraries needed:

pip install kaggle
pip install numpy
pip install lightgbm
pip install sklearn
pip install matplotlib
pip install seaborn

cd scripts
python simple_model_with_all_tables_training.py

python simple_model_with_all_tables_scoring.py



Jar to create PPML file from model file
using build 1.2.14 from https://github.com/jpmml/jpmml-lightgbm

java -jar jpmml-lightgbm-executable-1.2.14.jar --lgbm-input ../model/gbm_classifier.txt --pmml-output ../model/gbm_classifier.pmml
