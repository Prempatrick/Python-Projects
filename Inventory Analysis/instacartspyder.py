# Instacart Dataset 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import matplotlib as mpl

warnings.filterwarnings('ignore')


pd.set_option('display.float_format', lambda x: '%.3f' % x )

from subprocess import check_output

print(check_output(["F:\Inventory Analysis\Inventory case studies\Instacart\data"]).decode("utf-8"))

%matplotlib inline 
# Importing the data into pandas dataframe 

order_product_train= pd.read_csv("F:\Inventory Analysis\Inventory case studies\Instacart\data\order_products__train.csv")
order_product_prior= pd.read_csv("F:\Inventory Analysis\Inventory case studies\Instacart\data\order_products__prior.csv")

orders = pd.read_csv("F:\Inventory Analysis\Inventory case studies\Instacart\data\orders.csv")
products= pd.read_csv("F:\Inventory Analysis\Inventory case studies\Instacart\data\products.csv")

aisles = pd.read_csv(r"F:\Inventory Analysis\Inventory case studies\Instacart\data\aisles.csv")
department= pd.read_csv("F:\Inventory Analysis\Inventory case studies\Instacart\data\departments.csv")


# Exploring the order_product_train or order_product_prior datasets since in description they have specially mentioned 
# the these dataset contains the previous purchase records 


print('The Dataset order_product_train has {} rows and {} columns'.format(order_product_train.shape[0], order_product_train.shape[1]))
print('The Dataset order_product_prior has {} rows and {} columns'.format(order_product_prior.shape[0], order_product_prior.shape[1]))

#  Checking the head of the dataset 

order_product_prior.head()  # reorder is 0 or 1
order_product_train.head()  # reorder is 0 or 1


# Since the order_product_prior and order_product_train has same number of the columns we will 
# Concat them together 


order_products= pd.concat([order_product_prior,order_product_train], axis=0)


# Checking the missing Data 

total=order_products.isnull().sum().sort_values(ascending=False)
percentage= order_products.isnull().sum()/len(order_products)*100

# There is no missing value in the order_product dataset 



# Checking the unique types of the orders and products  

orders_unique= len(set(order_products.order_id))
print("There are {} unique order in orders_product dataset".format(orders_unique))
products_unique= len(set(order_products.product_id))
print("There are {} unique products in orders_product dataset".format(products_unique))


# The orders with highest units 

group=order_products.groupby('order_id')['add_to_cart_order'].agg('max').reset_index()
group= group.add_to_cart_order.value_counts()
sns.set_style('white')
fig, axes= plt.subplots(figsize=(15,12))
plt.xticks(rotation='vertical',fontsize=10)
plt.yticks(fontsize=12)
sns.barplot(group.index, group.values, palette="muted")
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Number of Items per order",fontsize=14)
plt.ylabel("Count of the orders", fontsize=14)
plt.title("Plot Showing number of products per each order", fontsize=14)
plt.show()



# Checking which product moves fast 

group_pt= order_products.groupby('product_id')['reordered'].agg({'transaction_counts': 'count'}).reset_index()

group_pt= pd.merge(group_pt,products[['product_id','product_name']], on='product_id', how='left' )

group_pt.sort_values('transaction_counts', ascending=False, inplace=True)

# fruits were mostly ordered, so we plot the top ten products 

fig,axes= plt.subplots(figsize=(10,6))
plt.xticks(rotation='vertical', fontsize=12)
sns.barplot(group_pt['product_name'][0:10], group_pt['transaction_counts'][0:10])
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Product Name", fontsize=14)
plt.ylabel("Order Counts", fontsize=14)
plt.show()



# Reordering Frequency 

group_ro = order_products.groupby('reordered')['product_id'].agg({'total_reordered':'count'}).reset_index()

# We now find the ratios 

group_ro['ratio %']= group_ro['total_reordered'].apply(lambda x:  x/sum(group_ro['total_reordered'])*100)




# Plotting the reordered ratio 

fig,axes= plt.subplots(figsize=(8,10))
g=sns.barplot(group_ro['reordered'], group_ro['ratio %'])
g.set_xticklabels({"First Time":0,"Reordered":1}, fontsize=15)
plt.title("The percentage of first time and reordered items" , fontweight='bold', fontsize=15)
plt.ylabel("Percentage", fontsize=15, fontweight='bold')
plt.xlabel('')


# 59% orders are reordered items 


# Most Reordered item

group_topro = order_products.groupby('product_id')['reordered']. agg({'reordered_count': 'count', 'reordered_sum': 'sum'}).reset_index()
group_topro['prob'] = group_topro['reordered_sum']/group_topro['reordered_count']
group_topro = pd.merge(group_topro,products[['product_id','product_name']], on='product_id', how='left' )

group_topro = group_topro[group_topro['prob']> 0.75].sort_values('prob', ascending=False)


fig, axes= plt.subplots(figsize=(12,4))
plt.xticks(rotation='vertical')
sns.barplot(group_topro.product_name[0:10], group_topro.prob[0:10])
plt.xlabel("Top Products among the ordered items",fontweight='bold',fontsize=15)
plt.ylabel("Probability of Reordereding the item")
plt.show()

# Raw veggie was most reordered item 


# Exploring the orders.csv 

orders.head()

#  Checking the missing values 

orders.isnull().sum()
percentage_missin= orders.isnull().sum()/len(orders)*100

# days_since_prior_order  has  206209  which is 6.2 percent of the total data 



# The hour of the day where most ordering happened 

group_dwo = orders.groupby('order_hour_of_day')['order_id'].agg({'tran_per_hours':'count'}).reset_index()

fig, axes= plt.subplots(figsize=(12,10))
sns.barplot(group_dwo.order_hour_of_day, group_dwo.tran_per_hours)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Number of Transaction on hourly basis")
plt.ylabel("Count of the Transaction")

# The bulk of the transaction happns between 7:00 am to 10:00 pm 

# The day of the week which has most transactions 

group_wod = orders.groupby('order_dow')['order_id'].agg({'tran_per_week':'count'}).reset_index()





# Plotting the same

fig, axes=plt.subplots(figsize=(12,10))
p=sns.barplot(group_wod.order_dow, group_wod.tran_per_week)
p.set_xticklabels({"Sunday":0,"Monday":1,"Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}, fontsize=15)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xlabel('')
plt.ylabel("Count of the Transaction", fontsize=13)




# Analysing the period of reorder of the items

grouped_reorder= orders.groupby('user_id')['days_since_prior_order'].agg({'avg_reorder': 'mean'}).reset_index()

grouped_reorder.sort_values('avg_reorder',ascending=False, inplace=True)

grouped_reorder['avg_reorder'].mean()


# The average days for the reordering is 15 Days 


# Checking the unique users in the whole data set 

print("The Unique Customers in the orders Datasets is {}".format (len(set(orders.user_id))))


# Let us analyse the product, department dataset and aisles dataset into main dataset called as item dataset. 

items= pd.merge(left=pd.merge(products,right=department, how='left'),right= aisles, how='left')



# The Department having highesdt number of products 

grp_dpt= items.groupby('department')['product_id'].agg({'products_department': 'count'}).reset_index().sort_values('products_department', ascending=False)
grp_dpt['ratio']= grp_dpt['products_department'].apply( lambda x: x/sum(grp_dpt['products_department'])).sort_values(ascending=False)



fig,axes= plt.subplots(figsize=(10,12))
plt.xticks(rotation='vertical')
sns.barplot(grp_dpt.department,grp_dpt.products_department)


# Most important aisle in each Department 
group_aisle= items.groupby(['department','aisle'])['product_id'].agg({'total_products': 'count'}).reset_index().sort_values('total_products',ascending=False)

fig,axes= plt.subplots(7,3,figsize=(20,45), gridspec_kw = dict(hspace=1.4))
for (aisle, group), ax in zip(group_aisle.groupby(["department"]), axes.flatten()):
     g=sns.barplot(group.aisle, group.total_products, ax=ax)
     ax.set(xlabel='Aisle', ylabel= 'Number of Product')
     g.set_xticklabels(labels=group.aisle,rotation=90,fontsize=12)
     


gd= group_aisle.groupby('department')['aisle'].agg({'count_pd':'count'}).reset_index().sort_values('count_pd',ascending=False)


# Performing the customer Segmentation 

order_prior= pd.merge(order_product_prior,orders, on=['order_id','order_id'])
order_prior= order_prior.sort_values(by=['user_id', 'order_id'])



# Performing

mt= pd.merge(order_product_prior,products, on= ['product_id','product_id'])
mt = pd.merge(mt,department,on=['department_id','department_id'])
mt= pd.merge(mt,aisles,on=['aisle_id','aisle_id'])


# Checking the unique aisles
len(mt['aisle'].unique())
 print("There are {} unique aisles in the dataset". format(len(mt['aisle'].unique())))


# 















# Now we check the most reordered products























 






