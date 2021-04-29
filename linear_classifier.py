import numpy as np
import matplotlib.pyplot as plt


filename = "iris.data"

with open(filename) as f:
  content = f.read().splitlines()

training_data = []
test_data = []
data = []

sample_size         = 50

end_training_data   = 50
start_training_data = 20

end_test_data   = 20
start_test_data = 0

n_features  = 4
n_classes   = 3

def grad_W_MSE_k (gk, tk, xk):
    return np.multiply(np.multiply((gk-tk),gk),(1-gk))*xk.T

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def split_data():
    for j in range(n_classes):
        for i in range(start_training_data + sample_size * j, end_training_data + sample_size * j):
            training_data.append(content[i].split(','))
            data.append(content[i].split(','))


    for j in range(n_classes):
        for i in range(start_test_data + sample_size * j, end_test_data + sample_size * j):

            lst = content[i].split(',')
            if lst[0] != '':
                test_data.append(lst)
                data.append(content[i].split(','))


    for sample in training_data:
        for i in range(n_features-1): 
            sample[i] = float(sample[i])

    for sample in test_data:
        for i in range(n_features-1): 
            sample[i] = float(sample[i])

    for sample in data:
        for i in range(n_features-1): 
            sample[i] = float(sample[i])

def classify(Weight, data):
        
    wrong = []
    correct = []
    for sample in data:
        sample_features = np.zeros((n_features+1, 1))
        
        for i in range(n_features):
            sample_features[i][0] = sample[i]
        sample_features[-1][0] = 1

        g = np.matmul(Weight, sample_features)

        label = np.argmax(g)
        if (label == 0):
            if sample[n_features] != 'Iris-setosa':
                sample.append(0)
                wrong.append(sample)
            else:
                correct.append(sample)

        elif (label == 1):
            if sample[n_features] != 'Iris-versicolor':
                sample.append(1)
                wrong.append(sample)
            else:
                correct.append(sample)

        elif (label == 2):
            if sample[n_features] != 'Iris-virginica':
                sample.append(2)
                wrong.append(sample)
            else:
                correct.append(sample)

    return wrong, correct;

def getTrueLabel(label):
    if label == 'Iris-setosa':
        return np.matrix([1,0,0]).T
    if label == 'Iris-versicolor':
        return np.matrix([0,1,0]).T
    if label == 'Iris-virginica':
        return np.matrix([0,0,1]).T

def update(Weight, alpha, wrong):

    for sample in wrong:
        
        sample_features = np.zeros((n_features+1, 1))
        for i in range(n_features):
            sample_features[i][0] = sample[i]
        sample_features[-1][0] = 1
        z = np.matmul(Weight, sample_features)   
        g = sigmoid(z)
        #print(np.argmax(sample_features))
        t = getTrueLabel(sample[n_features])
        G = grad_W_MSE_k(g, t, sample_features)
        #print(G[2])

        Weight = Weight -  alpha * grad_W_MSE_k(g, t, sample_features)
    return Weight

def train():

    W = np.zeros((n_classes, n_features+1))
    alpha = 0.01
    iterations = 0
    while True:
        
        wrong, correct = classify(W, training_data)
        W = update(W, alpha, wrong)
        iterations = iterations + 1
        if not wrong or iterations > 10000 or np.linalg.norm(W) > 10:
            print(iterations)
            print(np.linalg.norm(W))
            print(W)
            break

    return W, wrong, correct

def test(Weight):
    wrong, correct = classify(W, test_data)
    return wrong, correct;

def labelToNum(label):
    if label == 'Iris-setosa':
        return 0
    elif label == 'Iris-versicolor':
        return 1
    elif label == 'Iris-virginica':
        return 2

def confusion_matrix(misses, hits):
    predicton_per_class = np.zeros((n_classes,n_classes))
    for miss in misses:
        x = labelToNum(miss[n_features])
        y = miss[-1]
        predicton_per_class[x][y] =  predicton_per_class[x][y] + 1 

    for hit in hits:
        x = labelToNum(hit[n_features])
        y = labelToNum(hit[n_features])
        predicton_per_class[x][y] =  predicton_per_class[x][y] + 1   

    return predicton_per_class

def plot_histograms():

    num_cols = np.uint8(np.floor(np.sqrt(n_features)))
    num_rows = np.uint8(np.ceil(n_features / num_cols))

    fig = plt.figure(figsize=(5, 5))

    sample_class1 = data[0:sample_size]
    sample_class2 = data[sample_size:sample_size*2]
    sample_class3 = data[sample_size*2:sample_size*3]



    for feature_index in range(n_features):
        ax = fig.add_subplot(num_cols, num_rows, feature_index + 1)
        ax.set(xlabel='Measurement [cm]', ylabel='Number of samples')
        
        feature_class1 = []
        feature_class2 = []
        feature_class3 = []
        
        for i in range(sample_size):
            feature_class1.append(sample_class1[i][feature_index])
            feature_class2.append(sample_class2[i][feature_index])
            feature_class3.append(sample_class3[i][feature_index])


        ax.hist(feature_class1, alpha= 0.5, stacked=True, label = 1)
        ax.hist(feature_class2, alpha= 0.5, stacked=True, label = 2)
        ax.hist(feature_class3, alpha= 0.5, stacked=True, label = 3)


        ax.set_title(feature_index)
        ax.legend(prop={'size': 7})

    plt.show()

def removeFeature(feature_index):
    new_training_data = []
    new_test_data = []
    for sample in training_data:
        new_sample = []
        for i in range(n_features+2):
            if i != feature_index:
                new_sample.append(sample[i])

        new_training_data.append(new_sample)

    for sample in test_data:
        new_sample = []
        for i in range(n_features+2):
            if i != feature_index:
                new_sample.append(sample[i])

        new_test_data.append(new_sample)

    return new_training_data, new_test_data


split_data()
n_features = n_features - 1
training_data, test_data = removeFeature(1)
n_features = n_features - 1
training_data, test_data = removeFeature(0)
n_features = 1
training_data, test_data = removeFeature(0)

#n_features = n_features - 1
#training_data, test_data = removeFeature(0)


W, wrong, correct = train()
misses, hits = test(W)

print("\nError rate training: \t", len(wrong)/(len(training_data)))
print("Error rate test: \t", len(misses)/(len(test_data)))

print("\nConfusion matrix training set: \n", confusion_matrix(wrong,correct))
print("\nConfusion matrix test set: \n", confusion_matrix(misses,hits))