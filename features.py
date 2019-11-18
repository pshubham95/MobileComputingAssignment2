import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import fftpack,integrate
import numpy as np
from scipy.stats import kurtosis
from os import listdir
from flask import Flask, request

from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from scipy.stats.stats import pearsonr
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
def fft_feat(val):
    l = len(val)
    s = val.index.min()
    cgValues = val.to_numpy()
    cgValues = cgValues.ravel()
    cgValues = cgValues[::-1]

    cgmFFTValues = abs(fftpack.fft(cgValues[s:l]))
    sortedA = sorted(cgmFFTValues[2:len(cgmFFTValues)], reverse=True)
    # plt.stem(time[2:28], cgmFFTValues[2:28])
    #print(sortedA[0:3], np.array(sortedA).mean())
    t = sortedA[0:3]
    t.append(np.array(sortedA).mean())
    return t

def rms(array):
   return np.sqrt(np.mean(array ** 2))


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def Auc(val):
    #val = list(val)
    return [abs(integrate.simps(val, dx=5))]

def kurr(val):
    #val = list(val)
    return [kurtosis(val)]

def windowedRms(val):
    chunks = list(divide_chunks(list(val), 10))
    rms_arr = []
    for i in range(0, len(chunks)):
        rms_arr.append(rms(np.array(chunks[i])))
    return rms_arr



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def getFeatures(df):
    df['index'] = df.index
    # scaling values
    df['leftWrist_x'] = df['leftWrist_x'] - df['nose_x']
    df['leftWrist_y'] = df['leftWrist_y'] - df['nose_y']
    df['rightWrist_x'] = df['rightWrist_x'] - df['nose_x']
    df['rightWrist_y'] = df['rightWrist_y'] - df['nose_y']
    df['leftElbow_x'] = df['leftElbow_x'] - df['nose_x']
    df['leftElbow_y'] = df['leftElbow_y'] - df['nose_y']
    df['rightElbow_x'] = df['rightElbow_x'] - df['nose_x']
    df['rightElbow_y'] = df['rightElbow_y'] - df['nose_y']
    # plot values
    '''df.plot(x='index', y=['leftWrist_x', 'leftWrist_y'])
    df.plot(x='index', y=['rightWrist_x', 'rightWrist_y'])
    df.plot(x='index', y=['leftElbow_x', 'leftElbow_y'])
    df.plot(x='index', y=['rightElbow_x', 'rightElbow_y'])
    df.plot(x='index', y=['leftWrist_x', 'leftWrist_y'])
    plt.show()'''
    '''windowedRms(df['leftElbow_y']),
                            windowedRms(df['leftElbow_x']),
                            windowedRms(df['rightElbow_y']),
                            windowedRms(df['rightElbow_x']),
                            windowedRms(df['leftWrist_x']),
                            windowedRms(df['leftWrist_y']),
                            windowedRms(df['rightWrist_y']),
                            windowedRms(df['rightWrist_x']),'''
    feature_matrix = [
        fft_feat(df['leftWrist_x']),
        fft_feat(df['leftWrist_y']),
        fft_feat(df['rightWrist_x']),
        fft_feat(df['rightWrist_y']),

        Auc(df['leftWrist_x']),
        Auc(df['leftWrist_y']),
        Auc(df['rightWrist_y']),
        Auc(df['rightWrist_x']),
        # [spatial.distance.cosine(df['leftWrist_x'], df['leftWrist_y'])],
        # [spatial.distance.cosine(df['rightWrist_x'], df['rightWrist_y'])],
        # [pearsonr(df['leftWrist_x'], df['leftWrist_y'])[0]],
        # [pearsonr(df['rightWrist_x'], df['rightWrist_y'])[0]],

        [angle_between(np.array(df['rightWrist_x']), np.array(df['rightWrist_y']))],
        [angle_between(np.array(df['leftWrist_x']), np.array(df['leftWrist_y']))]

    ]
    return feature_matrix

def convertJsonToCsv(jsonData):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    data = jsonData
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    return pd.DataFrame(csv_data, columns=columns)

def getFeaturesDF():
    fields = [
        'score_overall',
        'leftElbow_score',
        'leftElbow_x',
        'leftElbow_y',
        'rightElbow_score',
        'rightElbow_x',
        'rightElbow_y',
        'leftWrist_score',
        'leftWrist_x',
        'leftWrist_y',
        'rightWrist_score',
        'rightWrist_x',
        'rightWrist_y',
        'nose_score',
        'nose_x',
        'nose_y'
    ]
    folders = ['book', 'car', 'gift', 'movie', 'sell', 'total']
    labels = {
        'book': 1,
        'car': 2,
        'gift': 3,
        'movie': 4,
        'sell': 5,
        'total': 6
    }
    label_arr = []
    training_data = []
    for i in range(0, len(folders)):
        onlyfiles = listdir("./CSV/data/" + folders[i] + "/")
        for j in range(0, len(onlyfiles)):
            # print('./CSV/data/'+folders[i]+'/'+onlyfiles[j])
            df = pd.read_csv('./CSV/data/' + folders[i] + '/' + onlyfiles[j], usecols=fields)
            feature_matrix = getFeatures(df)
            training_data.append(list(itertools.chain(*feature_matrix)))
            label_arr.append(labels[folders[i]])
    return [label_arr, training_data]

def trainForest():
    t = getFeaturesDF()
    training_data = t[1]
    label_arr = t[0]
    sc = StandardScaler()
    transformed_matrix = sc.fit_transform(training_data)
    joblib.dump(sc, './scalar/scalarForest', compress=True)
    print(transformed_matrix)
    n = 5
    pca = PCA(n_components=n)
    principalComponents = pca.fit(transformed_matrix)
    feat_matrix = pca.transform(transformed_matrix)
    joblib.dump(pca.components_, './eigenVect/pca_componentsForest')
    feat_matrix_pd = pd.DataFrame(feat_matrix)
    feat_matrix_pd['labels'] = label_arr
    feat_matrix_pd = shuffle(feat_matrix_pd)
    label_arr = list(feat_matrix_pd['labels'])
    #print(label_arr)
    del feat_matrix_pd['labels']
    feat_matrix_np = np.array(feat_matrix_pd)
    forestClassifier =  RandomForestClassifier(n_estimators=500, max_features='auto', n_jobs = -1,random_state =150)
    #print(list(feat_matrix_np))'''
    #mlpClassifier = SVC(kernel='rbf')
    forestClassifier.fit(feat_matrix_np, label_arr)
    joblib.dump(forestClassifier, './models/forestclassifier')
    
def testForest(content):
    pca_components = joblib.load('./eigenVect/pca_componentsForest')
    eigenValuesArray = np.array(pd.DataFrame(pca_components).T)
    sc = joblib.load('./scalar/scalarForest')
    df = convertJsonToCsv(content)
    df = pd.read_csv('./CSV/data/total/total_1_narvekar.csv')
    feat_matrix = [list(itertools.chain(*getFeatures(df)))]
    transformed_feature_matrix = sc.transform(feat_matrix)
    print(transformed_feature_matrix)
    testData = np.dot(transformed_feature_matrix[0], eigenValuesArray)
    forestModel = joblib.load('./models/forestclassifier')
    output = forestModel.predict([testData])
    return output

def trainMlp():
    t = getFeaturesDF()
    training_data = t[1]
    label_arr = t[0]
    sc = StandardScaler()
    transformed_matrix = sc.fit_transform(training_data)
    joblib.dump(sc, './scalar/scalar', compress=True)
    print(transformed_matrix)
    n = 5
    pca = PCA(n_components=n)
    principalComponents = pca.fit(transformed_matrix)
    feat_matrix = pca.transform(transformed_matrix)
    joblib.dump(pca.components_, './eigenVect/pca_components')
    feat_matrix_pd = pd.DataFrame(feat_matrix)
    feat_matrix_pd['labels'] = label_arr
    feat_matrix_pd = shuffle(feat_matrix_pd)
    label_arr = list(feat_matrix_pd['labels'])
    #print(label_arr)
    del feat_matrix_pd['labels']
    feat_matrix_np = np.array(feat_matrix_pd)
    mlpClassifier =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                       hidden_layer_sizes=(300, 100), random_state=150)
    #print(list(feat_matrix_np))'''
    #mlpClassifier = SVC(kernel='rbf')
    mlpClassifier.fit(feat_matrix_np, label_arr)
    joblib.dump(mlpClassifier, './models/mlpclassifier')


    '''kf = KFold(n_splits=50)
    scores = []
    for train_index, test_index in kf.split(feat_matrix_np):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(0 ,len(train_index)):
            X_train.append(feat_matrix_np[train_index[i]])
            y_train.append(label_arr[train_index[i]])
        for i in range(0, len(test_index)):
            X_test.append(feat_matrix_np[test_index[i]])
            y_test.append(label_arr[test_index[i]])
        svclassifier =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                       hidden_layer_sizes=(300, 100), random_state=150)
        svclassifier.fit(X_train, y_train)
        score = svclassifier.score(X_test, y_test)
        scores.append(score)'''
#trainMlp()
#print(np.array(scores).mean())

def testMlp(content):
    pca_components = joblib.load('./eigenVect/pca_components')
    eigenValuesArray = np.array(pd.DataFrame(pca_components).T)
    sc = joblib.load('./scalar/scalar')
    df = convertJsonToCsv(content)
    df = pd.read_csv('./CSV/data/total/total_1_narvekar.csv')
    feat_matrix = [list(itertools.chain(*getFeatures(df)))]
    transformed_feature_matrix = sc.transform(feat_matrix)
    print(transformed_feature_matrix)
    testData = np.dot(transformed_feature_matrix[0], eigenValuesArray)
    mlpModel = joblib.load('./models/mlpclassifier')
    output = mlpModel.predict([testData])
    return output

@app.route('/testModels', methods = ['POST'])
def testModels():
    content = request.get_json(silent=True)
    labels = {
        1: 'book',
        2: 'car',
        3: 'gift',
        4: 'movie',
        5: 'sell',
        6: 'total'
    }
    try:
        inp = input("1. MLP 2.Foredt")
        if inp==1:
            output = testMlp(content)
        else:
            output = testForest(content)
        op_dict = {
            1: labels[output[0]]
        }
    except Exception as e:
        return json.dumps({'error': str(e)}), 500
    return json.dumps(op_dict), 200
if __name__ == '__main__':
    trainMlp()
    trainForest()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
