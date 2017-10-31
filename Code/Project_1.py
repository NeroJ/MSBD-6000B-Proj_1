import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.cross_validation import KFold
import os
import tensorflow as tf

class Data:
    def __init__(self):
        self.data_dic = {}
        self.Feature = []
        self.Class = []
        self.testData = []
        # read traindata.csv
        try:
            self.data = open('traindata.csv')
            for line in self.data:
                line = line.replace('\n', '').split(',')
                self.Feature.append(line)
            self.Feature = np.array(self.Feature)
            self.data_dic['Feature'] = self.Feature
        except Exception, e:
            print 'Reading data error!'
            print e
        # read trainlabel.csv
        try:
            self.data = open('trainlabel.csv')
            for line in self.data:
                line = line.replace('\n', '').split(',')
                self.Class.append(line)
            self.Class = np.array(self.Class)
            self.data_dic['Class'] = self.Class
        except Exception, e:
            print 'Reading class error!'
            print e
        # read testdata.csv
        try:
            self.data = open('testdata.csv')
            for line in self.data:
                line = line.replace('\n', '').split(',')
                self.testData.append(line)
            self.testData = np.array(self.testData)
        except Exception, e:
            print 'Reading testdata error!'
            print e
    # data normalization
    def preDeal(self):
        self.X = preprocessing.normalize(self.data_dic['Feature'])
        self.Y = self.data_dic['Class']
    # five Fold testing
    def five_Fold(slef, model):
        kf = KFold(n=len(slef.Y), n_folds=5, shuffle=True)
        cv = 0
        average = []
        for tr, tst in kf:
            tr_features = slef.X[tr, :]
            tr_target = slef.Y[tr]
            tst_features = slef.X[tst, :]
            tst_target = slef.Y[tst]
            model.fit(tr_features, tr_target)
            tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
            tst_accuracy = np.mean(model.predict(tst_features) == tst_target)

            print "%d Fold Train Accuracy: %f, Test Accuracy: %f" % (cv, tr_accuracy, tst_accuracy)
            average.append(tst_accuracy)
            cv += 1
        ave = sum(np.array(average)) / 5
        print 'The average of Test Accuracy:' + str(ave)
    # training test data
    def training_Data(self, model):
        model.fit(self.X, self.Y)
        self.result = model.predict(self.testData)
    # create CSV
    def createCSV(self, fileName):
        self.path = os.path.abspath('.')
        self.path = self.path + '/'+ fileName
        try:
            with open(self.path, "wb") as file:
                for item in self.result:
                    file.write(item)
                    file.write('\n')
                file.close()
        except Exception, e:
            print e
    # close data flow
    def __del__(self):
        self.data.close()

if __name__ == '__main__':
    ob_data = Data()
    ob_data.preDeal()
    # candidate models-----------------
    SVM_model = svm.SVC()
    GS = GaussianNB()
    DT = tree.DecisionTreeClassifier()
    LR = LogisticRegression()
    #----------------------------------
    ob_data.five_Fold(LR)
    #ob_data.training_Data(GS)
    #ob_data.createCSV('project1_20453306')


