import argparse
import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from collections import Counter
import pickle
def getFiles(train, path):
    images = []
    
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            
            images.append(os.path.join(path, os.path.join(folder, file)))
             
    if(train is True):
        np.random.shuffle(images) 
    
    return images

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def get_tiny_image(image):
    size = 16
    image = cv2.resize(image,(size, size))
    image = (image - np.mean(image))/np.std(image)
    tiny_images = np.asarray(image.flatten())
    return tiny_images

def readImage(img_path):
    return cv2.imread(img_path, 0)
    

def vstackDescriptors(descriptor_list):
    descriptor_list = list(filter(lambda x:x is not None,descriptor_list))
    descriptors = np.concatenate(descriptor_list, axis=0).astype('float32')
    '''descriptors = np.array(descriptor_list[0])
        for descriptor in descriptor_list[1:-1]:        
            if not descriptor is None:
                descriptors = np.vstack((descriptors, descriptor)) '''

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters,n_init=20).fit(descriptors)
    with open('C:\\Users\\Asakiyaa\\Desktop\\vocab.pkl','wb')as f:
            pickle.dump(kmeans.cluster_centers_,f,protocol=pickle.HIGHEST_PROTOCOL)
    #return np.vstack(kmeans.cluster_centers_)
    return kmeans   

def extractFeatures(vocab, descriptor_list, image_count, no_clusters):
    im_features = []
    def change_None_to_zeros(lst):
        if lst is None:
            return np.zeros((1,128))
        else:
            return lst
    descriptor_list = list(map(change_None_to_zeros,descriptor_list))
    for i in range(image_count):
        
        dist = distance.cdist(vocab, descriptor_list[i], metric='euclidean')
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        im_features.append(hist_norm)
        
        '''if descriptor_list[i] is not None:
            for j in range(len(descriptor_list[i])):
                feature = descriptor_list[i][j]
                feature=np.reshape(feature,(1,-1))
                idx = kmeans.predict(feature)
                im_features[i][idx] += 1'''

    return np.asarray(im_features)

def nearest_classifier(im_features,train_labels,test_features,k=10):
    
    prediction_labels=[]
    dist = distance.cdist(test_features, im_features, metric='euclidean')
    #dist = distance.cdist(train_image_feats, test_image_feats, metric='euclidean')
    
    
    for each in dist:
        idx = np.argsort(each)
        probs=[train_labels[i] for i in idx[:k]]
        counter = Counter(probs)
        most_common_element = counter.most_common(1) 
        prediction_labels.append(most_common_element[0][0])
   
    return prediction_labels

def trainModel(path, no_clusters,method):
    images = getFiles(True, path)
    print("Train images path detected.")
    
    descriptor_list = []
    train_labels = []
    
    image_count = len(images)
    classes=os.listdir(path)
    for img_path in images:
        for class_ in classes:
            if class_ in img_path:
                train_labels.append(class_)
                break
        
        img = readImage(img_path)
        if method=='sift':
            sift = cv2.SIFT_create()
            des = getDescriptors(sift, img)
           # _,des=dsift(img, step=[1,1], fast=True)
        elif method=='tiny':
            des = get_tiny_image(img)
        descriptor_list.append(des)

    if method=='sift':
        descriptors = vstackDescriptors(descriptor_list)    
        print("Descriptors vstacked.")
        #kmeans = clusterDescriptors(descriptors, no_clusters)
        print("Descriptors clustered.")  
        with open("C:\\Users\\Asakiyaa\\Desktop\\vocab.pkl",'rb')as vocabfile:
            vocab=pickle.load(vocabfile)
        im_features = extractFeatures(vocab, descriptor_list, image_count, no_clusters)
        print("Images features extracted.")
        scale = StandardScaler().fit(im_features)        
        im_features = scale.transform(im_features)
        print("Train images normalized.")

        return vocab, scale, im_features,train_labels, classes
    elif method=='tiny':
        descriptor_list=np.asarray(descriptor_list)
        scale = StandardScaler().fit(descriptor_list)        
        im_features = scale.transform(descriptor_list)
        return scale, im_features,train_labels, classes


    
def testModel(path, scale, im_features,train_labels, no_clusters,  classes,method,vocab=None):
    test_images = getFiles(False, path)
    print("Test images path detected.")
    
    count = 0
    true = []
    descriptor_list = []
    if method=='sift':
        sift = cv2.SIFT_create() 

    for img_path in test_images:
        img = readImage(img_path)
        if method=='sift':
            des = getDescriptors(sift, img)
        elif method=='tiny':
            des=get_tiny_image(img)
        if(des is not None):
            count += 1
            descriptor_list.append(des)
            for i in classes:
                if i.lower() in img_path.lower():
                    true.append(i)
                    break
    
    if method=='sift':
        
        test_features = extractFeatures(vocab, descriptor_list, count, no_clusters)
        test_features = scale.transform(test_features)
        predict_labels= nearest_classifier(im_features,train_labels,test_features)
    
    elif method=='tiny':
        test_features=descriptor_list
        test_features = scale.transform(test_features)
        predict_labels= nearest_classifier(im_features,train_labels,test_features)
    print("Test images classified.")
    
    counts=0
    counts = sum([1 for i, j in zip(predict_labels, true) if i == j])
    numbers={}
    for class_ in classes:
        numbers[class_]=sum([1 for i in true if i == class_])
    accuracies=[]
    for num in range(15):
        accuracies.append(sum([1 for i, j in zip(predict_labels, true) if i == j and i==classes[num]])/numbers[classes[num]])
    print("accuracies",accuracies)    
    
    accuracy=counts/len(predict_labels)
       
    confusion = confusion_matrix(predict_labels,true)
    visualize_confusion_matrix(confusion,accuracy,classes)
    return predict_labels,accuracy
    
def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()
        
def execute(train_path, test_path, no_clusters,method):
    if method=='sift':
        vocab, scale, im_features ,train_labels,classes= trainModel(train_path, no_clusters,method)
        testModel(test_path,  scale, im_features, train_labels,no_clusters, classes,method,vocab)
    elif method=='tiny':
        scale, im_features,train_labels, classes= trainModel(train_path, no_clusters,method)
        testModel(test_path, scale, im_features, train_labels,no_clusters, classes,method)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action="store", dest="train_path",default="C:\\Users\\Asakiyaa\\Desktop\\Homework3\\train")
    parser.add_argument('--test_path', action="store", dest="test_path", default="C:\\Users\\Asakiyaa\\Desktop\\Homework3\\test")
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=100)
    method='sift'
    args =  vars(parser.parse_args())
    execute(args['train_path'], args['test_path'], int(args['no_clusters']),method)
    
    
