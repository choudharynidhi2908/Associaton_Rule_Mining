
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt

book_data = pd.read_csv("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Association_Rule_mining\\book.csv")

print(book_data.head())

x = apriori(book_data,min_support =0.005,max_len=3,use_colnames = True)
print(x)
print(x.sort_values('support',ascending= False,inplace=True))


plt.bar(x= list(range(1,11)),height = x.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),x.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
plt.show()

rules = association_rules(x,metric="lift",min_threshold = 1)
rules.head(20)
print(rules.sort_values('lift',ascending= False,inplace = True))

