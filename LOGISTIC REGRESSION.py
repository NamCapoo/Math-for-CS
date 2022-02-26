import csv
import random
import math
import numpy as np
import pandas as pd
def readcsv(file):
    data = [[] for i in range (14)]
    with open(file,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for i in range(14):
                data[i].append(float(row[i]))
    return data
def readcsv1(file):
    with open(file) as file:
        reader = csv.reader(file,delimiter=' ')
        l = [row for row in reader]
        data = [list(x) for x in zip(*l)]
    return data
# data[cot][hang]
def gom_data(data,data1):
    final_data = [[] for i in rang(14)]
    for i in range()
def normalize(array,minmax):   
    for i in range(13):
        for j in range(len(array[0])):
            array[i][j] = (array[i][j]-minmax[i][0]) / (minmax[i][1]-minmax[i][0])

def minmax(array):
    minmax = list()
    for i in range(13):
            val_min = min((array[i]))
            val_max = max((array[i]))
            minmax.append([val_min,val_max])
    return minmax
            
def split_data(data):
    array_0 = [[] for i in range(9)]
    array_1 = [[] for i in range(9)]
    for i in range(768):
        for j in range(9):
            if data[8][i]==0 and len(array_0[j]) <= 250:
                array_0[j].append(data[j][i])
            if data[8][i]==1 and len(array_1[j]) <= 250:
                array_1[j].append(data[j][i])
    return array_0,array_1

def train(array):
    array_train = [[] for i in range(14)]
    for i in range(240):
        for j in range(14):
            array_train[j].append(array[j][i])

    array_test = [[] for i in range(14)]
    for i in range(240,297):
        for j in range(14):
            array_test[j].append(array[j][i])
    return array_train,array_test

def Loss(w,array,b):
    ans = 0 
    for i in range(len(array[0])):
        a = sigmoid_activation(value_z(w, b, array, i))
        ans += (-array[8][i]*math.log(a,math.e) - (1 - array[8][i])*math.log(1-a,math.e))
    ans /= len(array[0])
    return ans

def Loss_class_weighted(w,array,b):
    ans = 0 
    for i in range(len(array[0])):
        a = sigmoid_activation(value_z(w,b,array,i))
        ans += (-0.8*array[8][i]*math.log(a,math.e) - 0.2*(1 - array[8][i])*math.log(1 - a,math.e))           
    ans /= len(array[0])
    return ans
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def value_z(w,b,array,n):
    sum = 0
    for i in range(8):
       sum += w[i]*array[i][n] 
    #print(len(array[0]))
    ans = sum + b
    return ans
def reliability(w, b,array):
    accuracy = 0 
    for i in range (50):
        a = sigmoid_activation(value_z(w, b, array, i))
        if a > 0.5:
            a = 1
        else:
            a = 0
        if a == array[8][i]:
            accuracy +=1
    return accuracy/50

def accuracy_count(w,b,array):
    accuracy = 0 
    for i in range (50):
        a = sigmoid_activation(value_z(w, b, array, i))
        if a > 0.5:
            a = 1
        else:
            a = 0
        if a == array[8][i]:
            accuracy +=1
    return accuracy

def train_data_1(w,b,train):
    array_grad = [ 0 for i in range (13)]
    grad_b = 0
    for i in range(240):
        a = sigmoid_activation(value_z(w, b, train, i))
        for j in range(13):
            array_grad[j] += (a - train[13][i])*(train[j][i])
        grad_b += (a - train[13][i])
    
    for i in range(8):
        array_grad[i] /= 240
    grad_b /= 240
    for i in range(8):
        w[i] = w[i] - 0.1*array_grad[i]
    b = b - 0.1 * grad_b
    return w,b    

def train_data_unbalance(w,b,array):
    array_grad = [ 0 for i in range (8)]
    grad_b = 0
    for i in range(len(array[0])):
        a = sigmoid_activation(value_z(w, b, array, i))
        for j in range(8):
            array_grad[j] += (a - array[8][i])*(array[j][i])
        grad_b += (a - array_0[8][i])
    
    for i in range(8):
        array_grad[i] /= 250
    grad_b /= 250
    for i in range(8):
        w[i] = w[i] - 0.1*array_grad[i]
    b = b - 0.1 * grad_b
    return w,b    
# hệ số A = 0.2, B = 0.8
def Weighted_train_data(w,b, xy): #train data với trọng số
    ans = [0 for i in range(8)] #[0,0,0,0,0,0,0,0,0,0,0]
    grad_b = 0 
    # sigma(400) đạo hàm loss trên wi
    for i in range(len(xy[0])): #cứ mỗi hàng trong bộ train của mình
        z = value_z(w,b,xy,i) # tính y mũ
        a = sigmoid_activation(z) # sidmoid của y mũ
        for j in range(8):
            ans[j] += (0.2*a-0.8*xy[8][i]+0.6*a*xy[8][i])*(xy[j][i])
        grad_b += (0.2*a-0.8*xy[8][i]+0.6*a*xy[8][i])
    for i in range(8):
        ans[i] /= 250
    grad_b /= 250
    for i in range(8):
        w[i] = w[i] - 0.1*ans[i]
    b = b - 0.1 * grad_b
    return w,b    

# In ma trận và tính các thông số
def print_confusion_matrix(matrix):
    ma_tb = {'Actualy Positive(1)': pd.Series([matrix[0][0],matrix[1][0]],index = ['Predict Positive(1)','Predict Negative(0)']),
             'Actualy Negative(0)': pd.Series([matrix[0][1],matrix[1][1]],index = ['Predict Positive(1)','Predict Negative(0)'])}
    print(pd.DataFrame(ma_tb))
def accuracy(matrix):
    tu = matrix[0][0] + matrix[1][1]
    mau = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
    return tu/mau

def precision(matrix):
    tu = matrix[0][0]
    mau = matrix[0][0] + matrix[1][0]
    return tu/mau

def recall(matrix):
    tu = matrix[0][0]
    mau = matrix[0][0] + matrix[0][1]
    return tu/mau

def f(matrix):
    f = 2*(precision(matrix)*recall(matrix))
    return f/(precision(matrix)+recall(matrix))

# Nhập dữ liệu và chuẩn 
data = readcsv('d:/processed.csv') # đường dẫn file 

data1 = readcsv1('d:/reprocessed.csv') 
for i in range(len(data1[0])):
    for j in range(14):
        data1[j][i] = float(data[j][i])
data2 = [[] for i in range (14)]

train_data,test_data = train(data)
minmax = minmax(train_data)

normalize(train_data,minmax)
normalize(test_data,minmax)
for i in range(240):
    if train_data[13][i] >=1:
        train_data[13][i] =1

#array_0, array_1 = split_data(data)c

#train_1,test_1 = train(array_1)

b = random.random()
w = [] 
w = np.random.random((8))

#########-----TEST ZONE-----#############
#L = 1
#for i in range(1000):
 #   w,b = train_data_unbalance(w, b,train_array_0, train_array_1)
  #  L = (Loss_unbalance(w,train_array_0,b))
   #   print(L)







#########################################
def Train_balance(w,b,train_data,test_data):
    L = 1 
    for i in range(1000):
        w,b = train_data_1(w, b, train_data)
        L = (Loss(w,train_data,b) + Loss(w,train_data,b))/2
        if L < 1e-3:
            break
    tin1 = reliability(w,b,test_data)
    tin2 = reliability(w,b,test_data)
    accuracy1 = accuracy_count(w,b,test_data) # true negative
    accuracy2 = accuracy_count(w,b,test_data) # true positive
    matrix = [[accuracy2, 50 - accuracy1],[50 - accuracy2, accuracy1]]
    
    print_confusion_matrix(matrix)
    print("Loss of model is: ",L)
    print("Accuracy: ", accuracy(matrix))
    print("Precision: ", precision(matrix))
    print("Recall: ",recall(matrix))
    print("F1: ", f(matrix))
    print("Phần trăm dữ liệu chính xác sau khi train so với thực tế là:" , ((tin1+tin2)/2)*100,"%")
    

def Train_Class_Weighted(w,b,train_0,train_1,test_0,test_1):
    L = 1 
    array = [[] for i in range(9)]
    for i in range(50):
        for j in range(9):
            array[j].append(train_1[j][i])
    for i in range(200):
        for j in range(9):
            array[j].append(train_0[j][i])
    for i in range(1000):
        w,b = Weighted_train_data(w, b, array)
        L = Loss_class_weighted(w,array,b)

    tin1 = reliability(w,b,test_0)
    tin2 = reliability(w,b,test_1)
    accuracy1 = accuracy_count(w,b,test_0) # true negative
    accuracy2 = accuracy_count(w,b,test_1) # true positive
    matrix = [[accuracy2, 50 - accuracy1],[50 - accuracy2, accuracy1]]
    
    print_confusion_matrix(matrix)
    print("Loss of model is: ",L)
    print("Accuracy: ", accuracy(matrix))
    print("Precision: ", precision(matrix))
    print("Recall: ",recall(matrix))
    print("F: ", f(matrix))
    print("Phần trăm dữ liệu chính xác sau khi train so với thực tế là:" , ((tin1+tin2)/2)*100,"%")


print("-------------------------BALANCE MODEL------------------------")
#Train_balance(w, b, train_data,test_data)
#print("-------------------------UNBALANCE MODEL-------------------------")
#Train_unbalance(w, b)
#print("-------------------------CLASS WEIGHTED MODEL-------------------------") 
#Train_Class_Weighted(w,b,train_0,train_1,test_0,test_1)




