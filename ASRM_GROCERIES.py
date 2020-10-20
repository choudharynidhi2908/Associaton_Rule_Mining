# conda install -c conda-forge mlxtend

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

groceries =[]

with open("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Association_Rule_Mining\\groceries.csv") as f:
 groceries = f.read()



groceries = groceries.split('\n')
#print(groceries)

groceries_list = []
for i in groceries:
	groceries_list.append(i.split(','))
#print(groceries_list)


all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter
item_frequencies = Counter(all_groceries_list)

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
#print(item_frequencies)

frequencies = list(reversed([i[0] for i in item_frequencies]))
#print(frequencies)


plt.bar(x = frequencies[0:5],height = list(range(0,5)),color='rgbkymc')
plt.xticks(list(range(0,5),),frequencies[0:5]);plt.xlabel("items")
plt.ylabel("Count")
#plt.show()

groceries_series = pd.DataFrame(pd.Series(groceries_list))

groceries_series = groceries_series.iloc[:9835,:]
#print(groceries_series)

groceries_series.columns = ["Transactions"]

# print(groceries_series)
x = groceries_series['Transactions'].str.join(sep='*').str.get_dummies(sep = '*')

#print(x.head())

supp = apriori(x,min_support = 0.05,max_len= 4,use_colnames = True)
#print(supp)


#print(supp.sort_values('support',ascending= False,inplace = False))
plt.bar(x= list(range(1,11)),height = supp.support[1:11],color = 'rgmwk')

#plt.show()

lift_rules = association_rules(supp,metric = 'lift',min_threshold = 1)

print(lift_rules.head)

print(lift_rules.sort_values('lift',ascending= True,inplace = False))

plt.bar(x=list(range(1,5)),height= lift_rules.lift[1:5],color = 'rgmwk')
plt.show()

