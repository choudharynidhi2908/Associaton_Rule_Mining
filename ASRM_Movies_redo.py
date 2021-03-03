import pandas as pd 
import matplotlib.pyplot as plt 
from mlxtend.frequent_patterns import apriori, association_rules  as asr 

movies = pd.read_csv("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Association_Rule_Mining\\my_movies.csv")
print(movies)
movies_data = movies.iloc[1:,5:]
print(movies_data.head())
support_rules = apriori(movies_data,min_support = 0.05,max_len = 3,use_colnames = True)
print(support_rules.head())

plt.bar(x=list(range(1,11)), height = support_rules.support[1:11],color ='rgmyk')
plt.xticks(list(range(1,11)));support_rules.itemsets[1:11]
#plt.show()


lift_rules = asr(support_rules,metric = 'lift', min_threshold =1)

print(lift_rules)


plt.bar(x=list(range(1,11)), height = lift_rules.lift[1:11],color ='rgmyk')
plt.xticks(list(range(1,11)));lift_rules.lift[1:11]
plt.show()
#########################################################################


import pandas as pd

X=pd.read_csv("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Association_Rule_Mining\\my_movies.csv")

df=X.iloc[:,5:15]

from mlxtend.frequent_patterns import apriori,association_rules
frequent_items=apriori(df, min_support=0.005, max_len=3,use_colnames = True)

frequent_items.sort_values('support',ascending=False,inplace=True)

rules=association_rules(frequent_items,metric='lift', min_threshold=1)

print(rules.sort_values('lift',ascending=False).head())