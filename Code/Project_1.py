import numpy as np
import tensorflow as tf
import time
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
            #print self.Class
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

class Training:
    def __init__(self, ob):
        self.steps = 3000
        self.path = os.path.abspath('.') + '/'+ 'project1_20453306'
        self.x_data = ob.X
        self.y_data = ob.Y
        #define the placeholder
        self.xs = tf.placeholder(tf.float32, [None, 57])
        self.ys = tf.placeholder(tf.float32, [None, 1])

    def add_layer(self, input, in_size, out_size, activation_function=None):

        # original weights
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        # biases
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # w*x+b
        W_mul_x_plus_b = tf.matmul(input, Weights) + biases
        # activating function
        if activation_function is None:
            output = W_mul_x_plus_b
        else:
            output = activation_function(W_mul_x_plus_b)
        return output

    def design_BPN(self):
        # two hiden layer
        self.hidden_layer = self.add_layer(self.xs, 57, 45, activation_function=tf.nn.relu)
        self.hidden_layer1 = self.add_layer(self.hidden_layer, 45, 45, activation_function=tf.sigmoid)
        # output layer
        self.prediction = self.add_layer(self.hidden_layer1, 45, 1, activation_function=tf.sigmoid)
        # nerual network parameters
        # loss function
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction), reduction_indices=[1]))
        # training process
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.loss)

    def Output_file(self, result):
        try:
            with open(self.path, "wb") as file:
                for item in result:
                    # print item
                    file.write(str(float(item)))
                    file.write('\n')
            file.close()
        except Exception, e:
            print e

    def run(self):
        # initialize
        self.design_BPN()
        self.init = tf.global_variables_initializer()
        # Session
        self.sess = tf.Session()
        # initializing
        self.sess.run(self.init)
        # begins
        for i in range(self.steps):
            start_time = time.time()
            self.sess.run(self.train_step, feed_dict={self.xs: self.x_data, self.ys: self.y_data})
            if i % 100 == 0:
                lo, pre = self.sess.run([self.loss, self.prediction], feed_dict={self.xs: self.x_data, self.ys: self.y_data})
                print lo
                # print pre
        temp = self.sess.run(self.prediction, feed_dict={self.xs: self.x_data})
        aver = np.average(temp, axis=0)
        print aver
        result = [0 if i < aver + 0.05 else 1 for i in temp]
        result = np.array(result)
        cv = 0
        for i, j in zip(result, self.y_data):
            r = float(i) - float(j[0])
            if int(r) == 0:
                cv = cv + 1
        print (float(cv) / float(len(result)))

        temp = self.sess.run(self.prediction, feed_dict={self.xs: preprocessing.normalize(ob.testData)})
        aver = np.average(temp, axis=0)
        result = [0 if i < aver + 0.05 else 1 for i in temp]
        result = np.array(result)
        self.Output_file(result)

    def __del__(self):
        self.sess.close()

if __name__ == '__main__':
    ob = Data()
    ob.preDeal()
    T = Training(ob)
    T.run()
    # candidate models-----------------
    #SVM_model = svm.SVC()
    #GS = GaussianNB()
    #DT = tree.DecisionTreeClassifier()
    #RF = RandomForestClassifier(n_estimators=10)
    #LR = LogisticRegression()
    #----------------------------------
    #ob_data.five_Fold(RF)


