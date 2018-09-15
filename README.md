
DeepPhos
=========
DeepPhos: prediction of protein phosphorylation sites with deep learning
Developer: FenglinLuo  from Health Informatics Lab, School of Information Science and Technology, University of Science and Technology of China

Requirement
=========
keras==2.0.0
numpy>=1.8.0

Related data information need to first load
=========
test data.csv
test data contains 3 cols: proteinName, postion, sequence

predict
=========
You can change the corresponding parameters in  main function prdict.py to choose to use the model to predict for general or kinase prediction
The results is an txt file,like:
"Q99440"	"3"	"0.12992426753"
"Q99440"	"13"	"0.0967529118061"
"Q99440"	"19"	"0.101900868118"
"Q99440"	"33"	"0.786891698837"
"Q99440"	"42"	"0.830417096615"
"Q99440"	"60"	"0.0784499421716"

train 
=====
If you want to train your own network,your input file is an csv fie, while contains 4 columns:
label, proteinName, postion, sequence
You can change the corresponding parameters in  main function train.py to choose to use the model to predict for general or kinase prediction

Contact
Please feel free to contact us if you need any help: flluo@mail.ustc.edu.cn
