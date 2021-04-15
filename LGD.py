#!/usr/bin/env python
# coding: utf-8

# In[353]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[354]:


Data=pd.read_csv('LGD_Data.csv')


# In[355]:


pd.set_option('display.max_columns', 1000)


# In[356]:


Data.head()


# In[357]:


Data.columns.values.tolist()


# In[358]:


half_count = len(Data) / 2

Df=Data.dropna(axis=1,how='all', inplace=False, thresh=half_count)


# In[359]:


Df


# In[360]:


Df.columns


# In[361]:


Df.describe()


# In[362]:


Df.shape


# In[363]:


Df.isnull().sum()/466262*100


# In[364]:


Df.dtypes


# In[365]:


columns_to_delete = ['emp_title','url','sub_grade','issue_d','pymnt_plan','purpose','title','initial_list_status','last_pymnt_d','policy_code','last_credit_pull_d','Unnamed: 0','id','member_id']
Df.drop(columns_to_delete, inplace=True, axis=1)



# In[366]:


Df.columns


# In[367]:


Df.head(5)


# In[368]:


Df.info()


# In[369]:


Df.loan_amnt.unique()


# In[370]:


Df.loan_amnt.values.tolist()


# In[371]:


Df.info()


# In[372]:


Df.term.values


# In[373]:


Df.application_type.values.tolist()


# In[374]:


list_grades = ["A" , "B", "C", "D", "E", "F", "G"]
Df["grade"] = pd.Categorical(Df["grade"] , categories = list_grades, ordered = True)
Df.sort_values ('grade', inplace = True)

plt.subplot()
Df[Df["loan_status"]=="Fully Paid"]["installment"].hist(bins = 35, color = "blue", label = "loan_status = Fully Paid", alpha =0.6)
Df[Df["loan_status"]=="Charged Off"]["installment"].hist(bins = 35, color = "red", label = "loan_status = Charged Off", alpha = 0.6 )
plt.legend()
plt.xlabel("installment")


# In[375]:


Df['term']=Df['term'].astype('category')
Df['grade']=Df['grade'].astype('category')
Df['emp_length']=Df['emp_length'].astype('category')


# In[376]:


Df['emp_length'].value_counts()


# In[377]:


list_grades = ["< 1 year","1 year" , "2 years", "3 years", "4 years", "5 years", "6 years", "7 years","8 years","9 years","10+ years"]
Df["emp_length"] = pd.Categorical(Df["emp_length"] , categories = list_grades, ordered = True)
Df.sort_values ('emp_length', inplace = True)

plt.subplot()
Df[Df["loan_status"]=="Fully Paid"]["emp_length"].hist(bins = 35, color = "blue", label = "loan_status = Fully Paid", alpha =0.5)
Df[Df["loan_status"]=="Charged Off"]["emp_length"].hist(bins = 35, color = "red", label = "loan_status = Charged Off", alpha = 0.5)
plt.legend()
plt.xlabel("employement length")


# In[378]:


# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(Df['loan_amnt'], Df['recoveries'])
# set a title and labels
ax.set_title('DF Dataset')
ax.set_xlabel('c')
ax.set_ylabel('loan_amnt')


# In[379]:


# get correlation matrix
corr = Df.corr()
fig, ax = plt.subplots()
# create heatmap
im = ax.imshow(corr.values)

# set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),
                       ha="center", va="center", color="black")


# In[380]:


# we found that funded amount and the funded amount investement are highly corrolated


# In[381]:


sns.heatmap(Df.corr(), annot=True)


# In[382]:


columns_to_delete = ['loan_amnt','funded_amnt_inv']
Df.drop(columns_to_delete, inplace=True, axis=1)


# In[383]:


Df.zip_code.isnull().sum()


# In[384]:


Df["loan_status"].value_counts()


# In[385]:


meaning = [
"Loan has been fully paid off.",
"Loan for which there is no longer a reasonable expectation of further payments.",
"While the loan was paid off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
"While the loan was charged off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
"Loan is up to date on current payments.",
"The loan is past due but still in the grace period of 15 days.",
"Loan hasn't been paid in 31 to 120 days (late on the current payment).",
"Loan hasn't been paid in 16 to 30 days (late on the current payment).",
"Loan is defaulted on and no payment has been made for more than 121 days."]
status, count = Df["loan_status"].value_counts().index, Df["loan_status"].value_counts().values
loan_statuses_explanation = pd.DataFrame({'Loan Status': status,'Count': count,'Meaning': meaning})[['Loan Status','Count','Meaning']]
loan_statuses_explanation


# In[386]:


Df = Df[(Df["loan_status"] == "Fully Paid") |
(Df["loan_status"] == "Charged Off")]
mapping_dictionary = {"loan_status":{ "Fully Paid": 1, "Charged Off": 0}}
Df = Df.replace(mapping_dictionary)


# In[387]:


fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='loan_status',data=Df,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
Df.loan_status.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()


# In[388]:


print("Data types and their frequency\n{}".format(Df.dtypes.value_counts()))


# In[389]:


Df.recoveries.value_counts


# In[390]:


Df.recoveries.isnull().sum()


# In[391]:


Df['recoveries'] = Df['recoveries'].fillna(0)


# In[392]:


Df.recoveries.isnull().sum()


# In[393]:


Df['recoveries'] = Df.recoveries.astype(int)


# In[394]:


Df['total_pymnt'] = Df['total_pymnt'].fillna(0)


# In[395]:


Df['total_pymnt'] = Df['total_pymnt'].str.rstrip('%').astype('float')


# In[ ]:


Df['total_pymnt'].str.replace(r'credit_card','0').astype(float)

Df['total_pymnt'].str.replace(r'$170.00<br/>Pre-School ','0').astype(float)


# In[ ]:


Df['total_pymnt'].str.replace(r'credit_card','0').astype(float)


# In[ ]:



Df['total_pymnt'] = Df['total_pymnt'].str.rstrip('%').astype('float')


# In[ ]:



Df['total_pymnt'] = Df.total_pymnt.astype(int)
Df['total_pymnt_inv'] = Df.total_pymnt_inv.astype(int)
Df['total_rec_prncp'] = Df.total_rec_prncp.astype(int)
Df['total_rec_int'] = Df.total_rec_int.astype(int)
Df['total_rec_late_fee'] = Df.total_rec_late_fee.astype(int)
Df['collection_recovery_fee'] = Df.collection_recovery_fee.astype(int)
Df['last_pymnt_amnt'] = Df.last_pymnt_amnt.astype(int)
Df['collections_12_mths_ex_med'] = Df.collections_12_mths_ex_med.astype(int)
Df['tot_cur_bal'] = Df.tot_cur_bal.astype(int)
Df['total_rev_hi_lim'] = Df.total_rev_hi_lim.astype(int)


# In[ ]:


Df.dtypes


# In[ ]:





# In[ ]:




