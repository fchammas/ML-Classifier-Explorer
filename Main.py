import streamlit as st

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_score


import numpy as np
import matplotlib.pyplot as plt

st.title('Streamlit Example')
st.write("""
# Explore Different Classifier
which one is the best?
""")

data_sets = ("Iris","Breast Cancer", "Wine Dataset")

classifers = ("KNN", "SVM", "Random Forest", "Gaussian Naive Bayes")

dataset_name = st.sidebar.selectbox('Select Data Set', data_sets)
classifer_name = st.sidebar.selectbox('Select Classifier',classifers)

def load_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    
        
    X = data.data
    y = data.target
    return X, y

X, y = load_dataset(dataset_name)
st.write('Shape of the data set is: ', X.shape)
st.write('Number of classes is:', len(np.unique(y)))



def feature_selection(X,y):      
    f_number = st.sidebar.slider('Feature Selection', min_value=2, max_value= X.shape[1], value= X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=f_number)
    new_X = selector.fit_transform(X, y)
    st.write('Selected Features: ', new_X.shape[1])
    #st.write('Select Feautures Indices: ', selector.get_support(indices=True))
    return  new_X

X = feature_selection(X, y)
# now we want to split the data into train and test


splt = st.sidebar.slider('Split Percentage', 0.1,0.5, 0.2) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splt, random_state=1234)



# we want to make a function that return the parameter according to the classifier
def add_parameter_ui(clf_name):
    params = {}
    
    # knn
    if clf_name == "KNN":
        def best_k(X_train, X_test, y_train, y_test):
            errors = []
            for i in range(1, 16):
                knn = Pipeline(steps=[("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=i))])
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                error = zero_one_loss(y_test, y_pred, normalize=False)     
                errors.append(error)
            min_index = errors.index(min(errors))
            best_k_value = min_index + 1
            fig = plt.figure()
            plt.plot(range(1, 16), errors)
            plt.xlabel('K')
            plt.ylabel('Error')
            plt.title('Best K Value')
            st.pyplot(fig)
            
            
            
            return best_k_value
        
        k = best_k(X_train, X_test, y_train, y_test)
        st.write('Best K Value is: ', k)
        K = st.sidebar.slider('K', min_value=1, max_value=15, value=k)
        params['K'] = K
        st.write("K= ", k)
    # svm
    elif clf_name == "SVM":
        C = st.sidebar.slider('C', 0.01, 10.0)
        gamma = st.sidebar.slider("Gamma (Kernel Coefficient)", 0.0001, 10.0, 0.1, step=0.0001)
        kernel = st.sidebar.selectbox("Kernel Type", ["linear", "poly", "rbf", "sigmoid"], index=2)
        
        params['C'] = C
        params['gamma'] = gamma
        params['kernel'] = kernel
        
        
    # random forest  
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider('max_depth', min_value=2, max_value=100)
        n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
        
        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth
        
    # Gaussian Naive Bayes          
    elif clf_name == "Gaussian Naive Bayes":
        Smoothing = st.sidebar.slider('Smoothing', min_value=1e-9, max_value=1.0)
        params['Smoothing'] = Smoothing
        
    return params

params = add_parameter_ui(classifer_name)

def get_clssifer(clf_name, params):
    
    # knn
    if clf_name == "KNN":
        #clf = KNeighborsClassifier(n_neighbors=params['K'])       
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("KNN", KNeighborsClassifier(n_neighbors=params['K']))]
)
        
        
        
    # svm
    elif clf_name == "SVM":
        #clf  = SVC(C = params['C'] 
                #,gamma=  params['gamma']
        #           )
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("SVM", SVC(C = params['C']))]
        )
        
        
    # random forest
    elif clf_name == "Random Forest":
       # clf = RandomForestClassifier(
        #    n_estimators=params['n_estimators'], 
       #     max_depth=params['max_depth'], random_state=1234
        #)
        
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("Random Forest", RandomForestClassifier(n_estimators=params['n_estimators'],
                                                                                        max_depth=params['max_depth'], random_state=1234))]
        )
    # neural network
    elif clf_name == "Gaussian Naive Bayes":
        #clf = GaussianNB(var_smoothing=params['Smoothing'])
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("Gaussian Naive Bayes", GaussianNB(var_smoothing=params['Smoothing']))]
        )
    return clf

clf = get_clssifer(classifer_name, params)





# classification


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

#mean_zero_one_loss = zero_one_loss(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifer_name}")
st.write(f"Accuracy = {acc}")



def find_best_k(X, y):

    best_score = []  # Track the highest accuracy
    
    for k in range(1, 16):  # Loop through k values from 1 to 15
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5-Fold Cross Validation
        mean_score = np.mean(scores)  # Compute average accuracy
        
    
        best_score.append(mean_score)
            
    k = best_score.index(max(best_score)) + 1
    maximum = max(best_score)
    return k, maximum

k, max= find_best_k(X_train, y_train)
st.write(f"Best K: {k} with Accuracy: {max:.4f}")

#Plot
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.5, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)


# TODO
# add more parameters
# add other classifiers
# add features scaling
# add more data sets
