""" 
kogda nepravilno okrivaetsa data seperation/dc.duplicated().sum()
#sep = ';'
mojno udalit stolbec s to drop
del dc['']
stda zabolevanie peredayushimsa polovim putyem
""" 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report

dc = pd.read_csv("kag_risk_factors_cervical_cancer.csv") 
dc.info()
#zamena ? na nulya 
dc["First sexual intercourse"].replace({"?": "15.0"}, inplace=True)
dc["Number of sexual partners"].replace({"?": "3.0"}, inplace=True)
dc = dc.replace('?', "1.0") 

# convert datatype 
dc = dc.astype(float)
#dc = dc.rename(columns={'Dx:Cancer': 'Cancer', 'STDs:HIV': 'HIV'})

'''
s filna mojno napisat  u mena ne policilos 
data.fillna(data.mean(), inplace=True)
lino data=
column 22
'''

#sortirovka 
dc = dc.sort_index(ascending=True)
 
#udalit 'STDs:cervical condylomatosis', 'STDs:AIDS' cunki oxshar cavablar var ve balansi pozur
dc = dc.drop(columns = ['STDs:cervical condylomatosis', 'Dx'])


#dataset hagda statistik melumatlar 
ml = dc.describe()

#tak mojno otdelno po odnomu rassmatrivat znachenie 
dc.loc[222]
dc.loc[21]
dc.loc[22]
#v etom date net propushennix znacheniy
dc.isnull().sum()

#glavnie v date 
round(dc.mean(), 3)

fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,figsize=(20,40))
sns.countplot(x='Age', data=dc, ax=ax1)
sns.countplot(x='Number of sexual partners', data=dc, ax=ax2)
sns.countplot(x='Num of pregnancies', data=dc, ax=ax3)
sns.countplot(x='Smokes (years)', data=dc, ax=ax4)
sns.countplot(x='Hormonal Contraceptives (years)', data=dc, ax=ax5)
sns.countplot(x='IUD (years)', data=dc, ax=ax6)
sns.countplot(x='STDs (number)', data=dc, ax=ax7)

#matrica.S pomoshyu dannix sozdadim matricu tak rassmotrim danniy 
dc.corr()
corrMatrix = dc.corr()
sns.heatmap(corrMatrix, annot=True)

#pivot table
pt = pd.pivot_table(dc, index = 'Smokes (years)',
                     columns= 'Dx:Cancer',
                     values= 'Smokes', aggfunc= 'mean')
#t.k cancer ne zavisit  ot cancer

# Crosstable - saymaq üçün
crs = pd.crosstab(index = dc['Smokes'],
                   columns= dc['Dx:Cancer'])

crs2 = pd.crosstab(index = dc['STDs'],
                   columns= dc['Dx:Cancer'])

crs3 = pd.crosstab(index = [dc['STDs:AIDS'], dc ['STDs:HIV']],
                   columns= dc['Dx:Cancer'])

crs4 = pd.crosstab(index = dc['STDs:HPV'],
                   columns= dc['Dx:Cancer'])

crs5 = pd.crosstab(index = dc['STDs:HIV'], 
                   columns= dc['Dx:Cancer'])

crs6 = pd.crosstab(index = [dc['Dx:Cancer'], dc['Hormonal Contraceptives']],
                   columns= dc['Num of pregnancies'])


gh = dc[dc['Biopsy'] == 1] 
gh2 = dc[dc['Biopsy'] == 0]

#modeling
# Datalari X ve Ye bolmek
X = dc.iloc[:, :-1].values
Y = dc.iloc[:, -1].values


# Training Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)


''' Random Forest Classifier '''
cl = RandomForestClassifier(n_estimators= 5, criterion= 'entropy', random_state = 0)
cl.fit(X_train, Y_train)

# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
con = confusion_matrix(Y_test, y_pred)
#accuracy 
acrf = accuracy_score(Y_test, y_pred)

clrf = classification_report(Y_test, y_pred)

# Logistic Regression Model
cl = LogisticRegression()
cl.fit(X_train, Y_train)

# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
con2 = confusion_matrix(Y_test, y_pred)

acrf2 = accuracy_score(Y_test, y_pred)


clrf2 = classification_report(Y_test, y_pred)
'''
#tak mojno rassmotret skolko znachenie v dataset, naprimer u nas osnovnoy cancer ovet budet est ili net
print(dc['STDs:HIV'].unique())
print(dc['Biopsy'].unique())
print(dc['']) ''' 