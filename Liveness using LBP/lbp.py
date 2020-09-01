import cv2
from skimage import feature
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split


datapath = 'my_data'

def get_lbp(image,points,radius,tolerance):
	lbp = feature.local_binary_pattern(image,points,radius,method="uniform")
	hist, bins = np.histogram(lbp.ravel(),bins=np.arange(0,points + 3),range=(0,points + 2))
	hist = hist.astype("float")
	hist/= (hist.sum()+tolerance)

	return hist

datas= []
labels = []


for label in os.listdir(datapath):
	n = 1

	for image in os.listdir(os.path.join(datapath,label)):
		img_path = os.path.join(datapath,label,image)
		print(" Extracting lbp for image {} in {} class".format(n,label))

		img = cv2.imread(img_path)
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		hist = get_lbp(img_gray,24,8,1e-7)

		datas.append(hist)
		labels.append(label)

		n+=1

X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2,random_state=42) # 70% training and 30% test


model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


filename = 'finalized_model_Gradientboosting.sav'
pickle.dump(model, open(filename, 'wb'))
print("classifier saved...")



