#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn

# Statistical Quantitative Data:
# 
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

# Business Objective:
#     
# Actions taken to obtain qualitative data-
# In this project, we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers 
# at high risk of churn and identify the main indicators of churn.

# ## Step 1: Reading and Understanding the Data
# Let us first import NumPy and Pandas and read the day dataset of Telecom_Churn

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import necessary libraries to read the csv file
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import parallel_backend
from sklearn import metrics



import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import gc # for deleting unused variables
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


telecom=pd.read_csv('telecom_churn_data.csv')
telecom.head()


# In[4]:


telecom.shape


# In[5]:


telecom.info()


# In[6]:


telecom.describe()


# ## Step 2: Data Preprocessing

# #### Checking for null values

# In[7]:


# Checking the null values
telecom.isnull().values.sum()


# In[8]:


round(100*(telecom.loc[:,'mobile_number':'roam_og_mou_9'].isnull().mean()), 2)


# In[9]:


round(100*(telecom.loc[:,'loc_og_t2t_mou_6':'spl_og_mou_9'].isnull().mean()), 2)


# In[10]:


round(100*(telecom.loc[:,'og_others_6':'total_ic_mou_9'].isnull().mean()), 2)


# In[11]:


round(100*(telecom.loc[:,'spl_ic_mou_6':'date_of_last_rech_9'].isnull().mean()), 2)


# In[12]:


round(100*(telecom.loc[:,'last_day_rch_amt_6':'av_rech_amt_data_9'].isnull().mean()), 2)


# In[13]:


round(100*(telecom.loc[:,'vol_2g_mb_6':'sep_vbc_3g'].isnull().mean()), 2)


# In[14]:


# Let us remove those columns having null value percentage of more than 70%

telecom=telecom.drop(telecom.loc[:,'date_of_last_rech_data_6':'date_of_last_rech_data_9'].columns,axis=1)


# In[15]:


telecom=telecom.drop(telecom.loc[:,'arpu_3g_6' : 'night_pck_user_9'].columns,axis=1)


# In[16]:


telecom=telecom.drop(telecom.loc[:,'fb_user_6' : 'fb_user_9'].columns,axis=1)


# In[17]:


telecom.columns


# In[18]:


# Renaming the columns to format given in the dataset

telecom.rename(columns = {'jun_vbc_3g': 'vbc_3g_6',
                          'jul_vbc_3g': 'vbc_3g_7',
                          'aug_vbc_3g': 'vbc_3g_8',
                          'sep_vbc_3g':'vbc_3g_9'}, inplace = True)


# ### Imputation

# In[19]:


# Here we are imputing the null values with mode values in these columns

telecom['total_rech_data_6'].fillna(telecom['total_rech_data_6'].mode()[0], inplace=True)

telecom['total_rech_data_7'].fillna(telecom['total_rech_data_7'].mode()[0], inplace=True)

telecom['total_rech_data_8'].fillna(telecom['total_rech_data_8'].mode()[0], inplace=True)

telecom['total_rech_data_9'].fillna(telecom['total_rech_data_9'].mode()[0], inplace=True)


# In[20]:


# Here we are imputing the null values with mode values in these columns

telecom['max_rech_data_6'].fillna(telecom['max_rech_data_6'].mode()[0], inplace=True)

telecom['max_rech_data_7'].fillna(telecom['max_rech_data_7'].mode()[0], inplace=True)

telecom['max_rech_data_8'].fillna(telecom['max_rech_data_8'].mode()[0], inplace=True)

telecom['max_rech_data_9'].fillna(telecom['max_rech_data_9'].mode()[0], inplace=True)


# In[21]:


# Here we are imputing the null values with mode values in these columns

telecom['count_rech_2g_6'].fillna(telecom['count_rech_2g_6'].mode()[0], inplace=True)

telecom['count_rech_2g_7'].fillna(telecom['count_rech_2g_7'].mode()[0], inplace=True)

telecom['count_rech_2g_8'].fillna(telecom['count_rech_2g_8'].mode()[0], inplace=True)

telecom['count_rech_2g_9'].fillna(telecom['count_rech_2g_9'].mode()[0], inplace=True)


# In[22]:


# Here we are imputing the null values with mode values in these columns

telecom['count_rech_3g_6'].fillna(telecom['count_rech_3g_6'].mode()[0], inplace=True)

telecom['count_rech_3g_7'].fillna(telecom['count_rech_3g_7'].mode()[0], inplace=True)

telecom['count_rech_3g_8'].fillna(telecom['count_rech_3g_8'].mode()[0], inplace=True)

telecom['count_rech_3g_9'].fillna(telecom['count_rech_3g_9'].mode()[0], inplace=True)


# In[23]:


# Here we are imputing the null values because there exists mean values in these columns

telecom['av_rech_amt_data_6'].fillna(telecom['av_rech_amt_data_6'].mean(), inplace=True)

telecom['av_rech_amt_data_7'].fillna(telecom['av_rech_amt_data_7'].mean(), inplace=True)

telecom['av_rech_amt_data_8'].fillna(telecom['av_rech_amt_data_8'].mean(), inplace=True)

telecom['av_rech_amt_data_9'].fillna(telecom['av_rech_amt_data_9'].mean(), inplace=True)


# In[24]:


telecom.shape


# In[25]:


# As dataframe's rows of the columns having null values with less percentage are dropped
telecom=telecom.dropna()


# In[26]:


round(100*(telecom.isnull().mean()), 2)


# In[27]:


# Let us check whether telecom dataframe is having null value anymore
telecom.isnull().values.any()


# In[28]:


telecom.shape


# #### Checking for Outliers

# In[29]:


# Checking outliers at 25%, 50%, 75%, 90% , 95% and 1
telecom.describe(percentiles=[.25, .5, .75, .90, .95, 1])


# In[30]:


# We segregate the dataset in to object and non-object type to deal them accordingly
telecom_non_object=telecom.select_dtypes(exclude=['object'])
telecom_non_object


# In[31]:


telecom_non_object.columns


# In[32]:


telecom_non_object.dtypes


# In[33]:


telecom_non_object.shape


# In[34]:


# Remove columns having only '0' values

zero_col=[]
a=0
for col in telecom_non_object.columns:
    a=a+1
    count=0
    for row in telecom_non_object[col]:
        if row==0:
            count=count+1
        #count=0
    if len(telecom_non_object[col])==count:  
        zero_col.append(col)
    
#for col in telecom_non_object.columns:
    if len(telecom_non_object[col])==count:   
        telecom_non_object.drop([col],axis=1,inplace=True)

#print(\n)        
print('Total no. of columns',a,'\n')
#print(\n)
print('Columns having only zero values',zero_col,'\n')
print('no. of columns having only zero values',len(zero_col))


# In[35]:


telecom_non_object.shape


# In[36]:


telecom_non_object.describe()


# #### Dealing with outliers using quantiles

# In[37]:


telecom_non_object.describe(percentiles=[.25, .10, .70, .90, .95, 1])


# In[38]:


# We use 'quantiles' method to deal with outliers. We do have negative values and high positive values as the extreme
# values or outliers which should be removed.

low = .05
high = .95
quant_telecom = telecom_non_object.quantile([low, high])
print(quant_telecom)


# In[39]:


# Extracting the data seperating it from low and high quantiles or outliers
telecom_non_object = telecom_non_object.apply(lambda x: x[(x>quant_telecom.loc[low,x.name]) | 
                                    (x<quant_telecom.loc[high,x.name])], axis=0)


# In[40]:


telecom_non_object=telecom_non_object.drop(['mobile_number','circle_id'],axis=1)


# In[41]:


telecom_non_object.head()


# In[42]:


#telecom_non_object = pd.concat([telecom.loc[:,'mobile_number'], telecom_non_object], axis=1)
round((telecom_non_object['isd_og_mou_6'].isna().sum()/telecom_non_object.shape[0]),2)


# In[43]:


# Remove columns having only 'NaN' values

null_col=[]
a=0
for col in telecom_non_object.columns:
    a=a+1
    count=0
    for row in telecom_non_object[col]:
        if row==np.nan:
            count=count+1
        #count=0
    if round((telecom_non_object[col].isna().sum()/telecom_non_object.shape[0]),2)> 0.90:  
        null_col.append(col)
    
#for col in telecom_non_object.columns:
     
        telecom_non_object.drop([col],axis=1,inplace=True)

#print(\n)        
print('Total no. of columns',a,'\n')
#print(\n)
print('Columns having only nan values',null_col,'\n')
print('no. of columns having only nan values',len(null_col))


# In[44]:


telecom_non_object.shape


# In[45]:


telecom_non_object_50=telecom_non_object.iloc[:,:50]


# In[46]:


telecom_non_object_50.isna().sum()


# In[47]:


telecom_non_object_100=telecom_non_object.iloc[:,51:100]


# In[48]:


telecom_non_object_100.isna().sum()


# In[49]:


telecom_non_object_150=telecom_non_object.iloc[:,101:150]


# In[50]:


telecom_non_object_150.isna().sum()


# In[51]:


telecom_non_object_175=telecom_non_object.iloc[:,151:175]


# In[52]:


telecom_non_object_175.isna().sum()


# In[53]:


telecom_non_object.columns


# In[54]:


telecom_non_object.columns


# In[55]:


telecom_non_object.head()


# In[56]:


telecom_non_object.describe()


# In[57]:


telecom_non_object.dropna()


# In[58]:


telecom_non_object.isna().values.sum()


# In[59]:


telecom_non_object.head()


# In[60]:


telecom_non_object


# In[61]:


telecom_merge= telecom_non_object.copy()


# In[62]:


telecom_merge.isnull().values.sum()


# In[63]:


telecom_merge.head()


# ## Step3: Deriving new features

# #### Creating 'Churn' variable

# In[64]:


telecom_merge.shape


# In[65]:


telecom_merge['total_ic_mou_9']


# We will convert the continuous variables to boolean variables and then converted to binary variables. This is because to perform logical operations among boolean variables is necessary.

# In[66]:


# Other than value 0, every value is converted to 'True' and if 0 then it is 'False'
telecom_merge.loc[(telecom_merge.total_ic_mou_9 != 0),'total_ic_mou_9']=True


# In[67]:


telecom_merge['total_ic_mou_9']=telecom_merge['total_ic_mou_9'].astype('bool')
telecom_merge['total_ic_mou_9']


# In[68]:


telecom_merge.loc[(telecom_merge.total_og_mou_9 != 0),'total_og_mou_9']=True


# In[69]:


telecom_merge['total_og_mou_9']=telecom_merge['total_og_mou_9'].astype('bool')
telecom_merge['total_og_mou_9']


# In[70]:


telecom_merge.loc[(telecom_merge.vol_2g_mb_9 != 0),'vol_2g_mb_9']=True


# In[71]:


telecom_merge['vol_2g_mb_9']=telecom_merge['vol_2g_mb_9'].astype(bool)
telecom_merge['vol_2g_mb_9']


# In[72]:


telecom_merge.loc[(telecom_merge.vol_3g_mb_9 != 0),'vol_3g_mb_9']=True


# In[73]:


telecom_merge['vol_3g_mb_9']=telecom_merge['vol_3g_mb_9'].astype(bool)
telecom_merge['vol_3g_mb_9']


# In[74]:


# As per the condition given in the problem statement, either incoming or outgoing and either 2g network or 3g
# network used by the customer

telecom_merge['Churn']=(telecom_merge.total_ic_mou_9 & telecom_merge.total_og_mou_9)&(telecom_merge.vol_2g_mb_9 & telecom_merge.vol_3g_mb_9)


# In[75]:


# Check the values of 'Churn' column
telecom_merge['Churn']


# In[76]:


# We convert boolean values to binary numbers 0 and 1

telecom_merge.loc[(telecom_merge.Churn != False),'Churn']=1


# In[77]:


# We convert the datatype of column 'Churn' to 'int' type

telecom_merge['Churn']=telecom_merge['Churn'].astype(int)
telecom_merge['Churn']


# So it seems that there is a high class imbalance with 86%-non churners and 14%-churners are there. we deal it with appropriate operations.

# In[78]:


# As we have created new variable from 'total_ic_mou_9','total_og_mou_9','vol_2g_mb_9' and 'vol_3g_mb_9', we drop
# these variables which are of no use.

telecom_merge=telecom_merge.drop(['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9'],axis=1)


# In[79]:


print(telecom_merge.columns.tolist())


# In[80]:


telecom_merge.shape


# In[81]:


# We remove the column with suffix '_9'

telecom_merge = telecom_merge.drop([column for column in telecom_merge.columns if '_9' in column],axis=1)


# In[82]:


telecom_merge.shape


# In[83]:


# we don't need this variable
#telecom_merge.drop(['mobile_number'],axis=1,inplace=True)


# #### Deriving new (categorical) features using qcut

# In[84]:


telecom_merge.describe(percentiles=[.25, .50, .70, .99])


# In[85]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_loc_og_t2t_6'] = pd.qcut(telecom_merge['loc_og_t2t_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_og_t2t_7'] = pd.qcut(telecom_merge['loc_og_t2t_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_og_t2t_8'] = pd.qcut(telecom_merge['loc_og_t2t_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[86]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_loc_og_t2m_6'] = pd.qcut(telecom_merge['loc_og_t2m_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_og_t2m_7'] = pd.qcut(telecom_merge['loc_og_t2m_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_og_t2m_8'] = pd.qcut(telecom_merge['loc_og_t2m_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[87]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_total_og_6'] = pd.qcut(telecom_merge['total_og_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_total_og_7'] = pd.qcut(telecom_merge['total_og_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_total_og_8'] = pd.qcut(telecom_merge['total_og_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[88]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_loc_ic_t2t_6'] = pd.qcut(telecom_merge['loc_ic_t2t_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_ic_t2t_7'] = pd.qcut(telecom_merge['loc_ic_t2t_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_ic_t2t_8'] = pd.qcut(telecom_merge['loc_ic_t2t_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[89]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_loc_ic_t2m_6'] = pd.qcut(telecom_merge['loc_ic_t2m_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_ic_t2m_7'] = pd.qcut(telecom_merge['loc_ic_t2m_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_loc_ic_t2m_8'] = pd.qcut(telecom_merge['loc_ic_t2m_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[90]:


bin_labels = ['Low', 'Medium', 'High']

telecom_merge['cat_total_ic_6'] = pd.qcut(telecom_merge['total_ic_mou_6'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_total_ic_7'] = pd.qcut(telecom_merge['total_ic_mou_7'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)

telecom_merge['cat_total_ic_8'] = pd.qcut(telecom_merge['total_ic_mou_8'],
                              q=[0, 0.35, 0.70, 0.99],
                              labels=bin_labels)


# In[91]:


telecom_merge.drop(['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8',
                   'loc_ic_t2m_mou_6','loc_ic_t2m_mou_7','loc_ic_t2m_mou_8',
                   'loc_ic_t2t_mou_6','loc_ic_t2t_mou_7','loc_ic_t2t_mou_8',
                    'total_og_mou_6','total_og_mou_7','total_og_mou_8',
                    'loc_og_t2m_mou_6','loc_og_t2m_mou_7','loc_og_t2m_mou_8',
                    'loc_og_t2t_mou_6','loc_og_t2t_mou_7','loc_og_t2t_mou_8'],axis=1,inplace=True)


# In[92]:


telecom_merge.head()


# In[93]:


# To create new features, we take average of the first two columns i.e, 6th and 7th months

col_group = telecom_merge.loc[ :, ["total_rech_amt_6", "total_rech_amt_7"]]

telecom_merge['Avg_total_rech_amt_6_7_months'] = col_group.mean(axis=1)

telecom_merge['Avg_total_rech_amt_6_7_months']


# In[94]:


# Investigating the customers whose recharge amount is more than 70% of the total recharge amount
# in 6th and 7th months

telecom_merge.Avg_total_rech_amt_6_7_months.describe(percentiles=[.25, .10, .70, .90, .95, .99])


# In[95]:


X = telecom_merge.Avg_total_rech_amt_6_7_months.quantile(0.70);

print("70th percentile: ",X);


# In[96]:


telecom_merge_high_value = telecom_merge[telecom_merge['Avg_total_rech_amt_6_7_months']>=X]


# In[97]:


telecom_merge_high_value.shape


# ### Create Dummy variables

# In[98]:


# Choose only categorical variables with prefix 'cat_'
telecom_merge_smp = telecom_merge_high_value[[column for column in telecom_merge_high_value.columns if 'cat_' in column]]


# In[99]:


print(telecom_merge_smp.columns.tolist())


# In[100]:


cat_list=telecom_merge_smp.columns.tolist()


# In[101]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy_telecom = pd.get_dummies(telecom_merge_smp[cat_list], drop_first=True)


# In[102]:


# Adding the results to the master dataframe
telecom_merge_hv_dum = pd.concat([telecom_merge_high_value, dummy_telecom], axis=1)


# #### Drop the repeated variables

# In[103]:


# We drop the parent variables that are repeating after the creation of their dummy variables
telecom_merge_hv_dum=telecom_merge_hv_dum.drop(cat_list,axis=1)


# In[104]:


telecom_merge_hv_dum.head()


# In[105]:


telecom_merge_hv_dum.shape


# In[106]:


telecom_comp=telecom_merge_hv_dum.copy()


# In[107]:


# We create a set to store unique variables/features

corr_features=set()
corr_matrix = telecom_comp.corr()


# In[108]:


for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.55:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)


# In[109]:


# Total number of correlated features
len(corr_features)


# In[110]:


# Total correlated features
print(corr_features)


# In[111]:


# We remove the features which are highly correlated with each other which are not useful for building model

telecom_comp.drop(labels=corr_features, axis=1, inplace=True)


# In[112]:


telecom_comp.shape


# In[113]:


pd.set_option('display.max_columns', None)

print(telecom_comp.columns.tolist())


# In[114]:


# As recommended, to take average of the first two months columns and we filter the data above 0.70 of avg recharge
# amount. This indicates that we can derive few columns from other remaining columns of 6th and 7th

# # As per the business's instructions, considering more than 2 months (good phase) will be a late decision to take action
# # So we take only two series and take average on it.

col_group_roam_ic_mou= telecom_comp.loc[ :, ["roam_ic_mou_6", "roam_ic_mou_7"]]

telecom_comp['roam_ic_mou_6_7_months'] = col_group_roam_ic_mou.mean(axis=1)

telecom_comp['roam_ic_mou_6_7_months']


# In[115]:


col_group_spl_ic_mou= telecom_comp.loc[ :, ["spl_ic_mou_6", "spl_ic_mou_7"]]

telecom_comp['spl_ic_mou_6_7_months'] = col_group_spl_ic_mou.mean(axis=1)

telecom_comp['spl_ic_mou_6_7_months']


# In[116]:


col_group_og_others= telecom_comp.reindex(columns =["og_others_6", "og_others_7"])

#tips_filtered = tips_df.reindex(columns = filtered_columns)
                                          
telecom_comp['og_others_6_7_months'] = col_group_og_others.mean(axis=1)

telecom_comp['og_others_6_7_months']


# In[117]:


col_group_loc_og_t2c_mou= telecom_comp.reindex(columns= ['loc_og_t2c_mou_6', 'loc_og_t2c_mou_7'])

telecom_comp['loc_og_t2c_mou_6_7_months'] = col_group_loc_og_t2c_mou.mean(axis=1)

telecom_comp['loc_og_t2c_mou_6_7_months']


# In[118]:


col_group_spl_ic_mou= telecom_comp.reindex(columns=['spl_ic_mou_6', 'spl_ic_mou_7'])

telecom_comp['spl_ic_mou_6_7_months'] = col_group_spl_ic_mou.mean(axis=1)

telecom_comp['spl_ic_mou_6_7_months']


# In[119]:


# Drop the parent variables of 6th and 7th months from which we derived new variables

label_67=[


'og_others_6',
'spl_ic_mou_6','spl_ic_mou_7',


'roam_ic_mou_6','roam_ic_mou_7',

'spl_ic_mou_6', 'spl_ic_mou_7',
'loc_og_t2c_mou_6', 'loc_og_t2c_mou_7'

]


# In[120]:


telecom_comp.drop(labels=label_67, axis=1, inplace=True)


# In[121]:


# Displaying the total columns
print(telecom_comp.columns.tolist())


# In[122]:


telecom_comp.shape


# In[123]:


telecom_numeric=telecom_comp.copy()
telecom_numeric


# In[124]:


telecom_numeric.isna().any()


# In[125]:


telecom_imp=telecom_numeric.copy()


# In[126]:


telecom_imp.head()


# In[127]:


print(telecom_imp.columns.tolist())


# In[128]:


imp_cols=telecom_imp.columns.tolist()


# In[129]:


telecom_imp=telecom_imp.loc[:,imp_cols]


# In[130]:


telecom_imp.shape


# In[ ]:





# ### Handling Imbalanced Data

# In[131]:


#Ploting barplot for target 

plt.figure(figsize=(10,6))
bar_graph = sns.barplot(telecom_imp['Churn'], telecom_imp['Churn'], palette='Set1', estimator=lambda x: len(x) / len(telecom_imp) )

#Anotating the graph
for i in bar_graph.patches:
        width, height = i.get_width(), i.get_height()
        x, y = i.get_xy() 
        bar_graph.text(x+width/2, 
               y+height, 
               '{:.0%}'.format(height), 
               horizontalalignment='center',fontsize=15)

#Setting the labels
plt.xlabel('Churn Rate', fontsize=14)
plt.ylabel('Percentage of Churn', fontsize=14)
plt.title('Percentage of Customers will or will not Churn', fontsize=16)


# In[132]:


# Putting feature variable to X
X = telecom_imp.drop(['Churn'], axis=1)


# In[133]:


X.head()


# In[134]:


# Putting response variable to y
y = telecom_imp['Churn']


# In[135]:


y.head()


# In[136]:


# # target variable distribution
100*(telecom_imp['Churn'].astype('int').value_counts()/len(telecom_imp.index))


# In[137]:


sns.countplot(y)


# In[138]:


# To handle imbalanced data here we use SMOTE Analysis

from imblearn.over_sampling import SMOTE
smote=SMOTE()
X_smote,y_smote=smote.fit_resample(X,y)


# In[139]:


X_smote.head()


# In[140]:


type(X_smote)


# In[141]:


y_smote.value_counts()


# In[142]:


type(y_smote)


# In[143]:


sns.countplot(y_smote)


# ## Step4: Train-Test Split

# In[144]:


from sklearn.model_selection import train_test_split


# In[145]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, train_size=0.7, test_size=0.3, random_state=100)


# ### Step4a: Performing PCA analysis for Dimensionality Reduction

# #### Scaling before PCA

# In[146]:


X_p_train=X_train.copy()
X_p_test=X_test.copy()


# In[147]:


X_p_train.shape


# In[148]:


y_pca_train=y_train.copy()
y_pca_test=y_test.copy()


# In[149]:


X_cov_matrix=X_p_train.cov()


# In[150]:


eig_vals, eig_vecs= np.linalg.eig(X_cov_matrix)
print('Eigen Vectors \n%s', eig_vecs)
print('\nEigen Values \n%s', eig_vals)


# In[151]:


eig_pairs= [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x:x[0],reverse=True)

print('Eigen Values in Descending Order: ')
for i in eig_pairs:
    print(i[0])


# In[152]:


tot=sum(eig_vals)
var_exp =[(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
print("Variance  captured  by each component is \n",var_exp)
print(40*'-')
print("Cumulative  variance  captured as we travel each component \n",cum_var_exp)


# In[ ]:





# In[153]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[154]:


X_p_sc_train = scaler.fit_transform(X_p_train)
X_p_sc_test = scaler.transform(X_p_test)


# In[155]:


df_X_p_sc_train=pd.DataFrame(data=X_p_sc_train,columns=X_p_train.columns)
df_X_p_sc_test=pd.DataFrame(data=X_p_sc_test,columns=X_p_test.columns)


# In[156]:


df_X_p_sc_train.describe()


# In[ ]:





# In[157]:


from sklearn.decomposition import PCA

pca = PCA()
df_X_pca_train = pca.fit_transform(df_X_p_sc_train)
df_X_pca_test = pca.transform(df_X_p_sc_test)


# In[158]:


def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    plt.scatter(xs ,ys, c = y_pca_test) #without scaling
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

#Call the function. 
myplot(df_X_pca_test[:,0:47], pca.components_) 
plt.show()


# In[159]:


explained_variance_all = pca.explained_variance_ratio_
print(explained_variance_all)


# In[160]:


plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)


# In[161]:


var_cumu = np.cumsum(pca.explained_variance_ratio_)


# In[162]:


#making the scree plot
fig = plt.figure(figsize=[12,8])
plt.vlines(x=45, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=1.0, xmax=47, xmin=0, colors="g", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative variance explained")
plt.xlabel("number of features")
plt.show()


# In[163]:


# Let us take components which can describe almost 95% variance
pca8 = PCA(n_components=8, random_state=42)
X_pca_train = pca8.fit_transform(X_p_sc_train)
X_pca_test = pca8.transform(X_p_sc_test)


# In[164]:


print(pd.DataFrame(pca8.components_,columns=X_p_train.columns,
                   index = ['PC-1','PC-2','PC-3','PC-4','PC-5',
                            'PC-6','PC-7','PC-8']))


# In[165]:


X_pca_train.shape


# In[166]:


explained_variance_8 = pca8.explained_variance_ratio_
print(explained_variance_8)


# In[167]:


plt.bar(range(1,len(explained_variance_8)+1), explained_variance_8)


# In[168]:


var_cumu = np.cumsum(explained_variance_8)


# In[169]:


#making the scree plot
plt.plot(range(1,len(var_cumu)+1), var_cumu)


# ### Step5a: Classification Model to predict

# #### Model 1: Random Classifier Model

# In[170]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_pca_train, y_pca_train)


# In[171]:


X_pca_train.shape


# #### Evaluation

# In[172]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[173]:


# Predicting the Test set results
y_pca_pred_test = classifier.predict(X_pca_test)


# In[174]:


c_m_test = confusion_matrix(y_pca_test, y_pca_pred_test)
print(c_m_test)


# In[175]:


rcm_acc_test= accuracy_score(y_pca_test, y_pca_pred_test)
print('Accuracy: ',rcm_acc_test)


# In[176]:


# roc auc
roc_auc_RFC_test=metrics.roc_auc_score(y_pca_test, y_pca_pred_test)
print("ROC AUC Random Forest Classifier ",roc_auc_RFC_test)


# In[177]:


results = pd.DataFrame({'Model':['Random Classifier for Test'],'Accuracy': [rcm_acc_test], 'ROC-AUC':[roc_auc_RFC_test]})
results = results[['Model', 'Accuracy', 'ROC-AUC']]
results


# #### Model 2: Decision Tree Classifier Model

# In[178]:


from sklearn.tree import DecisionTreeClassifier


# In[179]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_pca_train, y_pca_train)


# In[180]:


# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[181]:


#plotting tree with max_depth=3
dot_data = StringIO()  

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=None, 
                class_names=['No_Churn', "Churn"])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[182]:


#y_pca_pred_dt = dt.predict(X_pca_test).astype(int)
acc_decision_tree = dt.score(X_pca_train,y_pca_train) 
print('Accuracy of decision tree: ', acc_decision_tree)


# #### Evaluating model performance

# In[183]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[184]:


y_pca_test_pred=dt.predict(X_pca_test)


# In[185]:


acc_dtc_test=accuracy_score(y_pca_test, y_pca_test_pred)
print('Accuracy of decision tree: ',acc_dtc_test)


# In[186]:


c_dtc_m_test=confusion_matrix(y_pca_test, y_pca_test_pred)
print(c_dtc_m_test)


# In[187]:


# roc auc
roc_auc_DTC_test=metrics.roc_auc_score(y_pca_test, y_pca_test_pred)
print("ROC AUC Decision Tree Classifier ",roc_auc_DTC_test)


# Creating helper functions to evaluate model performance and help plot the decision tree

# In[188]:


def get_dt_graph(dt_classifier):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True,rounded=True,
                    feature_names=None,
                    class_names=['Churn', "No_Churn"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


# In[189]:


def evaluate_model(dt_classifier):
    print("Test Accuracy :", accuracy_score(y_pca_test, dt_classifier.predict(X_pca_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_pca_test, dt_classifier.predict(X_pca_test)))


# ##### Without setting any hyper-parameters

# In[190]:


dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_pca_train, y_pca_train)


# In[191]:


evaluate_model(dt_default)


# #### Controlling the depth of the tree

# In[192]:


dt_depth = DecisionTreeClassifier(max_depth=3)
dt_depth.fit(X_pca_train, y_pca_train)


# In[193]:


gph = get_dt_graph(dt_depth) 
Image(gph.create_png())


# In[194]:


evaluate_model(dt_depth)


# #### Hyper parameter Tuning

# In[195]:


dt = DecisionTreeClassifier(random_state=42)


# In[196]:


from sklearn.model_selection import GridSearchCV


# In[197]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[198]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=3, verbose=1, scoring = "accuracy")


# In[199]:


X_pca_train.shape


# In[200]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_pca_train, y_pca_train)')


# In[201]:


score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()


# In[202]:


score_df.nlargest(5,"mean_test_score")


# In[203]:


grid_search.best_estimator_


# In[204]:


dt_best = grid_search.best_estimator_


# In[205]:


evaluate_model(dt_best)


# In[206]:


from sklearn.metrics import classification_report


# In[207]:


print(classification_report(y_pca_test, dt_best.predict(X_pca_test)))


# In[208]:


gph = get_dt_graph(dt_best)
Image(gph.create_png())


# In[209]:


DTC=DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)


# In[210]:


DTC.fit(X_pca_train,y_pca_train)


# In[211]:


# Predicting on Test


# In[212]:


y_pca_pred_DTC_test = DTC.predict(X_pca_test).astype(int)


# In[213]:


acc_dtc_test=accuracy_score(y_pca_test, y_pca_pred_DTC_test)
print('Accuracy of decision tree for Test: ',acc_dtc_test )


# In[214]:


# roc auc
roc_auc_DTC_test=metrics.roc_auc_score(y_pca_test, y_pca_pred_DTC_test)
print("ROC AUC Random Forest Classifier for Test",roc_auc_DTC_test)


# In[215]:


tempResults = pd.DataFrame({'Model':['Decision Tree classifier for Test'], 
                            'Accuracy': [acc_dtc_test], 'ROC-AUC':[roc_auc_DTC_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy', 'ROC-AUC']]
results


# ### Step 5b: Logistic Regression Model through PCA

# In[216]:


from sklearn.linear_model import LogisticRegression


# In[217]:


logreg_pca=LogisticRegression(solver='lbfgs')


# In[218]:


logreg_pca.fit(X_pca_train,y_pca_train)


# In[219]:


y_pca_test_pred = logreg_pca.predict(X_pca_test)


# In[220]:


# Predicting on Test data


# In[221]:


logreg_pca.predict(X_pca_test)


# In[222]:


lor_pca_acc_test=logreg_pca.score(X_pca_test, y_pca_test)
print("Accuracy of logistic Regression(PCA)for Test: ",lor_pca_acc_test)


# In[223]:


c_logregpca_m_test=confusion_matrix(y_pca_test, y_pca_test_pred)
print(c_logregpca_m_test)


# In[224]:


# roc auc
roc_auc_logreg_test=metrics.roc_auc_score(y_pca_test, y_pca_test_pred)
print("ROC AUC under logreg(PCA) for Test",roc_auc_logreg_test)


# In[225]:


tempResults = pd.DataFrame({'Model':['Logistic regression(PCA) for Test'], 
                            'Accuracy': [lor_pca_acc_test], 'ROC-AUC':[roc_auc_logreg_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy', 'ROC-AUC']]
results


# ### Adaboost

# In[226]:


# adaboost classifier with max 600 decision trees of depth=2
# learning_rate/shrinkage=1.5

# base estimator
tree = DecisionTreeClassifier(max_depth=2)

# adaboost with the tree as base estimator
adaboost_model_1 = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")


# In[227]:


# fit
adaboost_model_1.fit(X_pca_train, y_pca_train)


# In[228]:


# predictions
# the second column represents the probability of a churning
predictions = adaboost_model_1.predict_proba(X_pca_test)
predictions[:10]


# In[229]:


# metrics: AUC
metrics.roc_auc_score(y_pca_test, predictions[:,1])


# #### Adaboost-Hyperparameter Tuning

# In[230]:


# parameter grid
param_grid = {"base_estimator__max_depth" : [2,5],
              "n_estimators": [200,400,600]
             }


# In[231]:


# base estimator
tree = DecisionTreeClassifier()

# adaboost with the tree as base estimator
# learning rate is arbitrarily set to 0.6

ABC = AdaBoostClassifier(
    base_estimator=tree,
    learning_rate=0.6,
    algorithm="SAMME")


# In[232]:


# run grid search
folds = 3
grid_search_ABC = GridSearchCV(ABC, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               n_jobs=8,
                               return_train_score=True,                         
                               verbose = 1)


# Note: The step below for grid_search takes approximately 15-20 mins to execute based on machine's processing speed

# In[233]:


# fit 
grid_search_ABC.fit(X_pca_train, y_pca_train)


# In[234]:


# cv results
cv_results = pd.DataFrame(grid_search_ABC.cv_results_)
cv_results


# In[235]:


# plotting AUC with hyperparameter combinations

plt.figure(figsize=(16,6))
for n, depth in enumerate(param_grid['base_estimator__max_depth']):
    

    # subplot 1/n
    plt.subplot(1,3, n+1)
    depth_df = cv_results[cv_results['param_base_estimator__max_depth']==depth]

    plt.plot(depth_df["param_n_estimators"], depth_df["mean_test_score"])
    plt.plot(depth_df["param_n_estimators"], depth_df["mean_train_score"])
    plt.xlabel('n_estimators')
    plt.ylabel('AUC')
    plt.title("max_depth={0}".format(depth))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# The results above show that:
# The ensemble with max_depth=5 is clearly overfitting (training auc is almost 1, while the test score is much lower)
# At max_depth=2, the model performs slightly better with a higher test score
# Thus, we should go ahead with max_depth=2 and n_estimators=600.
# Note that we haven't experimented with many other important hyperparameters till now, such as learning rate, subsample etc., and the results might be considerably improved by tuning them. We'll next experiment with these hyperparameters.

# In[236]:


grid_search_ABC.best_estimator_


# In[237]:


# model performance on test data with chosen hyperparameters

# base estimator
tree = DecisionTreeClassifier(max_depth=2)

# adaboost with the tree as base estimator
# learning rate is arbitrarily set, we'll discuss learning_rate below
ABC = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5),
    learning_rate=0.6,
    n_estimators=600,
    algorithm="SAMME")

ABC.fit(X_pca_train, y_pca_train)


# In[238]:


# predict on test data
ABC_predictions_test = ABC.predict_proba(X_pca_test)
ABC_predictions_test[:10]


# In[239]:


ABC_pca_acc_tes=ABC.score(X_pca_test, y_pca_test)
print("Accuracy of Adaboost(PCA)for Test: ",ABC_pca_acc_tes)


# In[240]:


# roc auc
roc_auc_ABC_test=metrics.roc_auc_score(y_pca_test, ABC_predictions_test[:, 1])
print("ROC AUC under Adaboost for Test",roc_auc_ABC_test)


# In[241]:


tempResults = pd.DataFrame({'Model':['Adaboost(PCA) for Test'], 'Accuracy': [ABC_pca_acc_tes],
                            'ROC-AUC':[roc_auc_ABC_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy','ROC-AUC']]
results


# ### Gradient Boosting Classifier

# In[242]:


# parameter grid
param_grid = {"learning_rate": [0.2, 0.6, 0.9],
              "subsample": [0.3, 0.6, 0.9]
             }


# In[243]:


# adaboost with the tree as base estimator
GBC = GradientBoostingClassifier(max_depth=2, n_estimators=200)


# In[244]:


# run grid search
folds = 3
grid_search_GBC = GridSearchCV(GBC, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               n_jobs=8,
                               return_train_score=True,                         
                               verbose = 1)

grid_search_GBC.fit(X_pca_train, y_pca_train)


# In[245]:


cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()


# In[246]:


# # plotting
plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# It is clear from the plot above that the model with a lower subsample ratio performs better, while those with higher subsamples tend to overfit.
# Also, a lower learning rate results in less overfitting.

# In[247]:


grid_search_GBC.best_estimator_


# In[248]:


GBC=GradientBoostingClassifier(learning_rate=0.2,
                               n_estimators=200,
                               subsample=0.6,
                               max_depth=2)


# In[249]:


GBC.fit(X_pca_train,y_pca_train)


# In[250]:


# predict on test data
GBC_predictions_test = GBC.predict_proba(X_pca_test)
GBC_predictions_test[:10]


# In[251]:


GBC_pca_acc_tes=GBC.score(X_pca_test, y_pca_test)
print("Accuracy of GradientBoost(PCA)for Test: ",GBC_pca_acc_tes)


# In[252]:


# roc auc
roc_auc_GBC_test=metrics.roc_auc_score(y_pca_test, GBC_predictions_test[:, 1])
print("ROC AUC under GradientBoost for Test",roc_auc_GBC_test)


# In[253]:


tempResults = pd.DataFrame({'Model':['GradientBoost(PCA) for Test'],  
                            'Accuracy': [GBC_pca_acc_tes], 'ROC-AUC':[roc_auc_GBC_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy','ROC-AUC']]
results


# ### XGBoosting Classifier

# In[254]:


# fit model on training data with default hyperparameters
XGBmodel = XGBClassifier()
XGBmodel.fit(X_pca_train, y_pca_train)


# In[255]:


# make predictions for test data
# use predict_proba since we need probabilities to compute auc
y_xgb_pred = XGBmodel.predict_proba(X_pca_test)
y_xgb_pred[:10]


# In[256]:


# evaluate predictions
roc = metrics.roc_auc_score(y_pca_test, y_xgb_pred[:, 1])
print("AUC: %.2f%%" % (roc * 100.0))


# In[257]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {"base_estimator__max_depth" : [2, 5],
              'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model_exp = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
xgbmodel_cv = GridSearchCV(estimator = xgb_model_exp, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        n_jobs=8,
                        verbose = 1,
                        return_train_score=True)      


# In[258]:


# fit the model
xgbmodel_cv.fit(X_pca_train, y_pca_train)       


# In[259]:


# cv results
cv_results = pd.DataFrame(xgbmodel_cv.cv_results_)
cv_results


# In[260]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_max_depth'] = cv_results['param_base_estimator__max_depth'].astype('float')
cv_results.head()


# In[261]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[262]:


xgbmodel_cv.best_estimator_


# In[263]:


# chosen best hyperparameters from the above xgbmodel_cv model
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
           'n_jobs':4,
          'subsample':0.3,
         'objective':'binary:logistic'}


# In[264]:


# fit model on training data
xgb_model = XGBClassifier(params = params)
xgb_model.fit(X_pca_train, y_pca_train)


# In[265]:


# plot
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.show()


# #### Predictions on Test Data

# In[266]:


# predict
XGB_predictions_test = xgb_model.predict_proba(X_pca_test)
XGB_predictions_test[:10]


# In[267]:


XGB_pca_acc_test=xgb_model.score(X_pca_test, y_pca_test)
print("Accuracy of XGBoost(PCA)for Test: ",XGB_pca_acc_test)


# In[268]:


# roc_auc
xgb_auc_test = metrics.roc_auc_score(y_pca_test,XGB_predictions_test[:, 1])
xgb_auc_test


# In[269]:


tempResults = pd.DataFrame({'Model':['XGBoost(PCA) for Test'],  
                            'Accuracy': [XGB_pca_acc_test], 'ROC-AUC':[xgb_auc_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy','ROC-AUC']]
results


# ### Feature Selection through RFE Method

# ### Feature Scaling

# In[270]:


from sklearn.preprocessing import StandardScaler


# In[271]:


X_log_train=X_train.copy()
X_log_test=X_test.copy()


# In[272]:


# X_train_s=X_train.copy()
# X_test_s=X_test.copy()


# In[273]:


col=X_train.columns.tolist()
col


# In[274]:


y_log_train = y_train.copy()
y_log_test = y_test.copy()


# In[275]:


### Checking the Churn Rate
churn = (sum(y_smote)/len(y_smote.index))*100
print("Churn Rate: ",churn)


# We have almost 50% churn rate after SMOTE ANALYSIS

# ### 6.a Feature Selection Using RFE

# #### K-Fold technique to deal with class imbalance

# In[276]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logreg = LogisticRegression()


# In[277]:


# creating a KFold object with 3 splits 
folds = KFold(n_splits = 3, shuffle = True, random_state = 100)

# specify range of hyperparameters
hyper_params = [{'n_features_to_select': list(range(2,48))}]

# specify model

logreg.fit(X_log_train, y_log_train)
rfe = RFE(logreg)             

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds,
                        n_jobs=5,
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_log_train, y_log_train)                  


# In[278]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[279]:


# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')


# In the above graph, we choose optimal number as 24, because after applying RFE() method the selection gradually reduces to significant number, to identify the important features. So,here the optimal number features obtained using KFold,GridSearchCV and RFE techniques is 24.

# In[280]:


# # final model
from sklearn import metrics as sme
n_features_optimal = 24

logreg = LogisticRegression()
logreg.fit(X_log_train, y_log_train)

rfe = RFE(logreg, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_log_train, y_log_train)


# In[281]:


# RFE supported/non-supported boolean values
rfe.support_


# In[282]:


# Below is the list of columns with RFE support expressed in terms of boolean values and also ranks suggested by RFE
list(zip(X_log_train.columns, rfe.support_, rfe.ranking_))


# In[283]:


col_rfe_support = X_log_train.columns[rfe.support_]


# In[284]:


# The below are the columns selected by RFE procedure
col_rfe_support


# ### 6.b Correlation Matrix

# In[285]:


# Let's copy X_l_train_s dataframe's columns to 'Sample' dataframe to check correlation among their features
Sample=X_log_train[col_rfe_support]


# In[286]:


# Let's see the correlation matrix 
plt.figure(figsize = (30,15))        # Size of the figure
sns.heatmap(Sample.corr(),annot = True)


# ### Dropping highly correlated  features

# In[287]:


# We create a set to store unique variables/features

corr_rfe_features=set()
corr_rfe_matrix = Sample.corr()


# In[288]:


for i in range(len(corr_rfe_matrix.columns)):
    for j in range(i):
        if abs(corr_rfe_matrix.iloc[i, j]) > 0.40:
            colname = corr_rfe_matrix.columns[i]
            corr_rfe_features.add(colname)


# In[289]:


# Total number of correlated features
len(corr_rfe_features)


# In[290]:


# Total correlated features
print(corr_rfe_features)


# In[291]:


# We remove the features which are highly correlated with each other which are not useful foe building model

Sample.drop(labels=corr_rfe_features, axis=1, inplace=True)


# In[292]:


Sample.shape


# In[293]:


col_non_corr_features=Sample.columns.tolist()
col_non_corr_features


# In[294]:


sca= StandardScaler()


# In[295]:


X_log_train=X_log_train[col_non_corr_features]


# ### 6.c Assessing the model with Statsmodel

# In[296]:


import statsmodels.api as sm
X_log_train_sm = sm.add_constant(X_log_train)
logm = sm.GLM(y_log_train,X_log_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# ### checking VIF values for features obtained from RFE

# In[297]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[298]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_log_train.columns
vif['VIF'] = [variance_inflation_factor(X_log_train.values, i) for i in range(X_log_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[299]:


results_summary = res.summary()
results_as_html = results_summary.tables[1].as_html()
res_df=pd.read_html(results_as_html, header=0, index_col=0)[0]


# In[300]:


res_df


# In[301]:


res_df.index.values


# In[302]:


res_df['P>|z|'].idxmax()


# In[303]:


res_df['P>|z|'].values


# In[304]:


res_list=res_df['P>|z|'].values.tolist()
res_list


# In[305]:


for i in range(len(res_list)):
    if res_list[i]>0.05:
        res_df.drop(res_df['P>|z|'].idxmax(),inplace=True)
        break


# In[306]:


res_df.index.values


# In[307]:


res_df.index[1:,]


# In[308]:


X_log_train=X_log_train[res_df.index[1:,]]
X_log_train


# In[309]:


X_log_train.columns


# In[310]:


#print(X_log_train.columns.tolist())


# In[311]:


#X_log_train.shape


# ### Assessing the models with StatsModels

# In[312]:


X_train_sm = sm.add_constant(X_log_train)
logm = sm.GLM(y_log_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[313]:


#Dropping the feature with high p-value
#X_log_train.drop(['og_others_6_7_months'],axis=1,inplace=True)


# In[314]:


results_summary = res.summary()
results_as_html = results_summary.tables[1].as_html()
res_df=pd.read_html(results_as_html, header=0, index_col=0)[0]


# In[315]:


res_df['P>|z|'].idxmax()


# In[316]:


res_list=res_df['P>|z|'].values.tolist()
res_list


# In[317]:


for i in range(len(res_list)):
    if res_list[i]>0.05:
        res_df.drop(res_df['P>|z|'].idxmax(),inplace=True)
        break


# In[318]:


res_df.index.values


# In[319]:


res_df.index[1:,]


# In[320]:


X_log_train=X_log_train[res_df.index[1:,]]
X_log_train


# In[321]:


X_log_train.columns


# In[ ]:





# In[322]:


X_train_sm = sm.add_constant(X_log_train)
logm = sm.GLM(y_log_train,X_train_sm, family = sm.families.Binomial())
res = logm.fit()
res.summary()


# In[323]:


# Selected Features
col_zero_pvalue=X_log_train.columns.tolist()


# In[324]:


X_log_train.shape


# In[325]:


X_log_test=X_log_test[col_zero_pvalue]


# In[326]:


# Check the shape of X_l_test_s_2 dataframe
X_log_test.shape


# In[327]:


# X_rfe_train is used for boosting classifiers

X_rfe_train=X_log_train.copy()


# In[328]:


# X_rfe_test is used for boosting classifiers

X_rfe_test=X_log_test.copy()


# In[329]:


# y_rfe_train and y_rfe_test is used for boosting classifiers

y_rfe_train = y_train.copy()
y_rfe_test = y_test.copy()


# In[330]:


# Check the shape of X_log_test dataframe
X_log_test.shape


# In[331]:


X_test_sm = sm.add_constant(X_log_test)
logm2 = sm.GLM(y_log_test,X_test_sm, family = sm.families.Binomial())
res2 = logm2.fit()
res2.summary()


# In[332]:


results_summary = res2.summary()
results_as_html = results_summary.tables[1].as_html()
res_df=pd.read_html(results_as_html, header=0, index_col=0)[0]


# In[333]:


res_list=res_df['P>|z|'].values.tolist()
res_list


# In[334]:


for i in range(len(res_list)):
    if res_list[i]>0.05:
        res_df.drop(res_df['P>|z|'].idxmax(),inplace=True)
        break


# In[335]:


res_df['P>|z|'].idxmax()


# In[336]:


res_df.index.values


# In[337]:


res_df.index[1:,]


# In[338]:


X_log_test=X_log_test[res_df.index[1:,]]
X_log_test


# In[339]:


X_log_test.columns


# In[ ]:





# In[ ]:





# In[340]:


#Dropping the feature with high p-value
# X_log_test.drop(['spl_og_mou_6'],axis=1,inplace=True)


# In[341]:


X_test_sm = sm.add_constant(X_log_test)
logm2 = sm.GLM(y_log_test,X_test_sm, family = sm.families.Binomial())
res2 = logm2.fit()
res2.summary()


# In[342]:


results_summary = res2.summary()
results_as_html = results_summary.tables[1].as_html()
res_df=pd.read_html(results_as_html, header=0, index_col=0)[0]


# In[343]:


res_list=res_df['P>|z|'].values.tolist()
res_list


# In[344]:


for i in range(len(res_list)):
    if res_list[i]>0.05:
        res_df.drop(res_df['P>|z|'].idxmax(),inplace=True)
        break


# In[345]:


res_df.index.values


# In[346]:


res_df.index[1:,]


# In[347]:


X_log_test=X_log_test[res_df.index[1:,]]
X_log_test


# In[348]:


test_cols=X_log_test.columns
X_log_test.columns


# In[349]:


X_test_sm = sm.add_constant(X_log_test)
logm2 = sm.GLM(y_log_test,X_test_sm, family = sm.families.Binomial())
res2 = logm2.fit()
res2.summary()


# In[350]:


col_zero_or_least_pvalue=X_log_test.columns.to_list()
col_zero_or_least_pvalue


# In[351]:


X_log_train=X_log_train[col_zero_or_least_pvalue]


# In[352]:


X_train_sm = sm.add_constant(X_log_train)
logm3 = sm.GLM(y_log_train,X_train_sm, family = sm.families.Binomial())
res3 = logm2.fit()
res3.summary()


# In[353]:


Final_columns=X_log_train.columns
Final_columns


# In[354]:


X_log_train.head()


# In[355]:


X_log_test=X_log_test[Final_columns]


# In[356]:


# Getting the predicted values on the train set
y_pred = res2.predict(X_test_sm)
y_pred[:10]


# #### Creating a dataframe with the actual convert flag and the predicted probabilities

# We create Churn Probability column from 'Churn' and predicted columns to determine customers who may churn in the "Churn phase" 

# In[357]:


y_pred_final = pd.DataFrame({'Churn':y_log_test.values, 'Churn_Prob':y_pred})
y_pred_final.head()


# In[358]:


y_pred_final.Churn_Prob.mean()


# #### Creating new column 'predicted' with 1 if Churn_Prob > 0.50321 else 0

# In[359]:


y_pred_final['predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.50321 else 0)

# Let's see the head
y_pred_final.head()


# In[360]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.predicted )
print(confusion)


# In[361]:


# Let's check the overall accuracy.
print("Accuracy Score: ",metrics.accuracy_score(y_pred_final.Churn, y_pred_final.predicted))


# In[362]:


### Checking the Churn Rate
churn = (sum(y_pred_final.predicted)/len(y_pred_final['predicted'].index))*100
print("Churn Rate: ",churn)


# ### 6.d Plotting ROC curve

# In[363]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[364]:


fpr, tpr, thresholds = metrics.roc_curve( y_pred_final.Churn, 
                                         y_pred_final.predicted, drop_intermediate = False )


# In[365]:


draw_roc(y_pred_final.Churn, y_pred_final.predicted)


# In[366]:


# Let's create columns with different Churn probability cutoffs 
Churn_prob = [float(x)/10 for x in range(10)] 
for i in Churn_prob:
    y_pred_final[i]= y_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_pred_final.head()


# In[367]:


# Now let's calculate accuracy sensitivity and specificity for various Churn probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['Churn_Prob','accuracy','sensitivity','specitivity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives


num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:
    cm1 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final[i])
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    specitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specitivity]

print(cutoff_df)


# In[368]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='Churn_Prob', y=['accuracy','sensitivity','specitivity'])
plt.show()


# In[369]:


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

threshold = Find_Optimal_Cutoff(y_pred_final.Churn, 
                                         y_pred_final.Churn_Prob)
print("Threshold Value: ", threshold)


# #### From the curve above, 0.39 is the optimum point to take it as a cutoff probability

# In[370]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > round(threshold[0],2) else 0)

y_pred_final.head()


# In[371]:


#Ploting barplot for target 
plt.figure(figsize=(10,6))
bar_graph = sns.barplot(y_pred_final['predicted'], y_pred_final['final_predicted'], palette='Set1', 
                        estimator=lambda x: len(x) / len(y_pred_final) )

#Annotating the graph
for p in bar_graph.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        bar_graph.text(x+width/2, y+height, '{:.0%}'.format(height), horizontalalignment='center',fontsize=15)

#Setting the labels for the graph
plt.xlabel('Churn Probability', fontsize=14)
plt.ylabel('Percentage of Churn', fontsize=14)
plt.title('Predictions of Customers who will or will not Churn', fontsize=16)


# The bar graph between 'Percentage of Churn' and 'Churn Probability' says that 54% customers are satisfied with network but whereas 46% of customers are about to churn because they might be dissatisified with the network or thinking to leave the network because of other network's better offers. But we have to say that all this graph happening in the 'Good Phase' and 'Action Phase'. If the Network team focus on the 46% customers and can retain them through variuos plans and offers then network can be successful overcoming this issue.

# ### 6.f Gradient Boosting through RFE

# In[372]:


X_rfe_train=X_log_train.loc[:,Final_columns]


# In[373]:


X_rfe_trafin=X_log_train.loc[:,Final_columns]


# In[374]:


X_rfe_test=X_log_test.loc[:,Final_columns]


# In[375]:


X_rfe_tesfin=X_log_test.loc[:,Final_columns]


# In[376]:


X_rfe_train=sca.fit_transform(X_rfe_train)
X_rfe_test=sca.transform(X_rfe_test)


# In[377]:


# parameter grid
param_grid = {"learning_rate": [0.2, 0.6, 0.9],
              "subsample": [0.3, 0.6, 0.9]
             }


# In[378]:


# Gradientboost with the tree as base estimator
GBC_rfe = GradientBoostingClassifier(max_depth=2, n_estimators=200)


# In[379]:


# run grid search
folds = 3
grid_search_GBC_rfe = GridSearchCV(GBC_rfe, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               n_jobs=8,
                               return_train_score=True,                         
                               verbose = 1)

grid_search_GBC_rfe.fit(X_rfe_train, y_rfe_train)


# In[380]:


cv_results = pd.DataFrame(grid_search_GBC_rfe.cv_results_)
cv_results.head()


# In[381]:


# # plotting
plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[382]:


grid_search_GBC_rfe.best_estimator_


# In[383]:


GBC_rfe=grid_search_GBC_rfe.best_estimator_

GBC_rfe.fit(X_rfe_train,y_rfe_train)


# In[384]:


# predict on test data
rfe_GBC_predictions_test = GBC_rfe.predict_proba(X_rfe_test)
rfe_GBC_predictions_test[:10]


# In[385]:


rfe_GBC_rfe_acc_tes=GBC_rfe.score(X_rfe_test, y_rfe_test)
print("Accuracy of GradientBoost(RFE)for Test: ",rfe_GBC_rfe_acc_tes)


# In[386]:


# roc auc
rfe_roc_auc_GBC_test=metrics.roc_auc_score(y_rfe_test, rfe_GBC_predictions_test[:, 1])
print("ROC AUC under GradientBoost for Test",rfe_roc_auc_GBC_test)


# In[387]:


tempResults = pd.DataFrame({'Model':['GradientBoost(RFE) for Test'],  
                            'Accuracy': [rfe_GBC_rfe_acc_tes], 'ROC-AUC':[rfe_roc_auc_GBC_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy','ROC-AUC']]
results


# ### 6.g XG Boosting through RFE

# In[388]:


# fit model on training data with default hyperparameters
xgb_rfe_model = XGBClassifier()
xgb_rfe_model.fit(X_rfe_train, y_rfe_train)


# In[389]:


# make predictions for test data
# use predict_proba since we need probabilities to compute auc
y_rfe_pred_test = xgb_rfe_model.predict_proba(X_rfe_test)
y_rfe_pred_test[:10]


# In[390]:


# evaluate predictions
roc_test = metrics.roc_auc_score(y_rfe_test, y_rfe_pred_test[:, 1])
print("AUC: %.2f%%" % (roc_test * 100.0))


# In[391]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {"base_estimator__max_depth" : [2, 5],
              'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_rfe_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
xgb_rfe_model_cv = GridSearchCV(estimator = xgb_rfe_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        n_jobs=5,
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      


# In[392]:


# fit the model
xgb_rfe_model_cv.fit(X_rfe_train, y_rfe_train)


# In[393]:


# cv results
rfe_cv_results = pd.DataFrame(xgb_rfe_model_cv.cv_results_)
rfe_cv_results


# In[394]:


# convert parameters to int for plotting on x-axis
rfe_cv_results['param_learning_rate'] = rfe_cv_results['param_learning_rate'].astype('float')
rfe_cv_results['param_max_depth'] = rfe_cv_results['param_base_estimator__max_depth'].astype('float')
rfe_cv_results.head()


# In[395]:


# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[396]:


xgb_rfe_model_cv.best_estimator_


# In[397]:


# fit model on training data
xgb_rfemodel = xgb_rfe_model_cv.best_estimator_#(params = params)
xgb_rfemodel.fit(X_rfe_train, y_rfe_train)


# In[398]:


# predict
rfe_XGB_predictions_test = xgb_rfemodel.predict_proba(X_rfe_test)
rfe_XGB_predictions_test[:10]


# In[399]:


rfe_XGB_acc_test=xgb_rfemodel.score(X_rfe_test, y_rfe_test)
print("Accuracy of XGBoost(RFE)for Test: ",rfe_XGB_acc_test)


# In[400]:


# roc auc
rfe_roc_auc_XGB_test=metrics.roc_auc_score(y_rfe_test, rfe_XGB_predictions_test[:, 1])
print("ROC AUC under XGBoost(RFE) for Test",rfe_roc_auc_XGB_test)


# In[401]:


tempResults = pd.DataFrame({'Model':['XGBoost(RFE) for Test'],  
                            'Accuracy': [rfe_XGB_acc_test], 'ROC-AUC':[rfe_roc_auc_XGB_test]})
results = pd.concat([results, tempResults])
results = results[['Model', 'Accuracy','ROC-AUC']]
results


# ## 7. Analysis and Answers

# Q) Which model is best and why? 
# 
# A)
# 
# As we can see the above table, we prefer XGBoosting Classifier is the best model when going through PCA or RFE feature engineering techniques. XGBoost acquires maximum area under curve as well as possess significant accuracy to explain features behaviour.

# We also observe that the GradientBoosting classifier is also best next to XGBoosting as it holds good Accuracy and ROC-AUC values. Here LogisticRegression through RFE is gaining more weight over Logistic Regression through PCA.

# In[402]:


# feature importance
importance = dict(zip(X_rfe_trafin.columns, xgb_rfemodel.feature_importances_))
importance


# In[403]:


features=list(importance.keys())
fvalues=list(importance.values())
fig=plt.figure(figsize=(10,5))

#creating the bar plot
plt.xticks(rotation='vertical')
plt.bar(features, fvalues, color='maroon', width=0.4)


# In[404]:


features=list(importance.keys())
features


# In[405]:


fvalues=list(importance.values())
fvalues


# In[406]:


fig=plt.figure(figsize=(10,5))
plt.bar(features,fvalues,color='blue',width=0.4)
plt.xticks(features,rotation='vertical')
plt.show()


# In[407]:


# All the final features displayed in descending order
sorted(importance.items(), key=lambda x:x[1],reverse=True)


# In[ ]:





# ## Saving the Model

# In[408]:


import pickle

#dump information to that file
pickle.dump(xgb_rfemodel,open('model.pkl','wb'))

#load a model
pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:





# Q) What are the important features produced by the 'best' model?
# 
# 
# A) 
# 
# As we are preferring XGBoosting classifier model over others,it is ranking the features that are most important for this project. As we can observe the above list of important features with their weights, and among them 'total_rech_data_6' i.e, total amount of data recharged by the customers in the 6th month i.e, "Action-Phase". This is very important feature because in the action phase customer behaviour says whether he/she will churn or not for the next month.The data usage by the customer shows his/her class in the network family.

# The next important and weighted feature is 'max_rech_data_6', maximum amount of data recharged by the customers in the 6th month,'monthly_2g_6' is the 2G network consumed by the customer in the 6th month of the year. 2G network is still used by the old, classic and high valued customers for their 'stick-up' interests and they are asset to the T-network family.

# Q) What proposal(s) you want to make before T-Network team? How you want to help them in retaining customers?
# 
# A)
# 
# T-Network is adviced to shed light on the above important features to carry out critical actions to retain high valued customers. For example, if we look at the most important features like 'total_rech_data_6','max_rech_data_6', if customers are allowed to consume data at lower prices or aged on network customers (customers who have been with T-network for long term) are given discounts on price plans with uninterrupted data provision may help network company to retain the valuable customers. 
# 
# If we observe carefully, maximum no.of features are of 6th month i.e 'Good Phase'. This means, we should have a very focussed observation on 6th month, start recording the customer's experiences at their early walk with the T-network.  And also behaviour of 'Medium' class customers in the 6th month and expecting their satisfaction and upgrading to 'High' Class.

# In[ ]:




