import pandas as pd #brings in the pandas
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('train.csv')
#df stores as a dataframe
print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.isnull().sum())
print(df.describe())
print(df["Pclass"].value_counts(normalize=True)*100)
print(df.groupby("Sex")["Survived"].mean())
print(df["Age"].describe())
df["Age"] = df["Age"].fillna(df["Age"].median())
print(df["Age"].isnull().sum())
print(df["Age"].describe())
print(df["Embarked"].value_counts())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
## S is the most common embarkation port
print(df["Embarked"].isnull().sum())
print(df["Cabin"].value_counts(dropna=False))
df["Deck"] = df["Cabin"].str[0]
print(df["Deck"].value_counts(dropna=False))
#we can later label Deck as unknown for analysis
print(df.isnull().sum())
print(df.groupby("Pclass")["Survived"].mean())
#step3
df["AgeGroup"] = pd.cut(df["Age"], bins = [0,12,18,35,60,100],
    labels=["child", "teen","young adult", "adult", "senior"])
print(df.groupby("AgeGroup", observed=True)["Survived"].mean()*100)
#What was the overall survival rate?
print(df.groupby(["Pclass","Sex"])["Survived"].mean()*100)
#Did women survive more than men?
print(df.groupby("Embarked")["Survived"].mean()*100)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
#Sibsp= sibling spouse count Parch = parents children count
# +1 means the passenger himself
print(df[["FamilySize", "SibSp", "Parch"]].head())
print(df.groupby("FamilySize")["Survived"].mean()*100)
print(df["FamilySize"].value_counts().sort_index())
#step4
#Visualization 1 — Survival Rate by Sex
survival_by_sex = df.groupby("Sex")["Survived"].mean()*100
survival_by_sex.plot(kind = "bar")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%) ")
plt.title("Survival by Sex")
plt.tight_layout()
plt.show()
#Visualization 2: Survival Rate by Passenger Class
survival_by_pclass = df.groupby("Pclass")["Survived"].mean()*100
survival_by_pclass.plot(kind="bar")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate (%) ")
plt.title("Survival by Pclass")
plt.tight_layout()
plt.show()
#Visualization 3 — Age Distribution
age_bins = pd.cut(df["Age"], bins = 20)
print(age_bins.value_counts())
df["Age"].plot(kind = "hist", bins = 20)
plt.xlabel("Age")
plt.ylabel("Number of passengers")
plt.title("Age distribution of titanic passengers")
plt.tight_layout()
plt.show()
##Visualization 4 — Survival by Age
survival_by_age = df.groupby("AgeGroup", observed=True)["Survived"].mean()*100
print(survival_by_age)
survival_by_age.plot(kind="bar")
plt.xlabel("Age")
plt.ylabel("Survival Rate (%) ")
plt.title("Survival by AgeGroup")
plt.tight_layout()
plt.show()
##Visualization 5 — Pclass + Sex survival
survival_by_pclass_sex = df.groupby(["Pclass", "Sex"])["Survived"].mean()*100
print(survival_by_pclass_sex)
survival_by_pclass_sex = survival_by_pclass_sex.unstack()
survival_by_pclass_sex.plot(kind="bar", color= ["salmon", "steelblue"])
plt.xlabel("Pclass")
plt.ylabel("Survival Rate (%) ")
plt.title("Survival by Pclass and Sex")
plt.legend(title = "Sex")
plt.tight_layout()
plt.show()
##Visualization 6 — FamilySize distribution
print(df["FamilySize"].value_counts().sort_index())
df["FamilySize"].value_counts().sort_index().plot(kind="bar", color= "salmon")
plt.xlabel("FamilySize")
plt.ylabel("Number of passengers")
plt.title("FamilySize Distribution")
plt.tight_layout()
plt.show()
##Visualization 7 — Correlation heatmap
corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]].corr()
sns.heatmap(corr, annot=True, cmap = "coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
