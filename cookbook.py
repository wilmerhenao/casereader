import numpy as np
import pandas as pd

url = ('https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv')

tips = pd.read_csv(url)
is_dinner = tips['time'] == 'Dinner'

s = pd.Series([1, 3, 5, np.nan, 6, 8])
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A': 1.,'B': pd.Timestamp('20130102'),'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df)
print(df.iloc[1:3])
print(df.iloc[1:3, :])
print(tips[['total_bill', 'tip', 'smoker', 'time']].head(5))
print(tips.loc[:,['total_bill', 'tip', 'smoker', 'time']].head(5))
print(tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5.00)])
print(tips.loc[(tips['time'] == 'Dinner') & (tips['tip'] > 5.00), ['total_bill', 'tip', 'smoker', 'time']])
#SELECT sex, count(*)
#FROM tips
#GROUP BY sex;
tips.groupby('sex').size()

#SELECT day, AVG(tip), COUNT(*)
#FROM tips
#GROUP BY day;
tips.groupby('day').agg({'tip': np.mean, 'day': np.size})
# Joining
pd.merge(left, right, on='key')
pd.merge(df1, df2, on='key', how='left') #or right or how='outer'

Boolean Indexing
df[df.A > 0]
df[df > 0] ; df2[df2 > 0] = -df2 # Setting in the second one
df2 = df.copy(); df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']; df2[df2['E'].isin(['two', 'four'])]

df1.dropna(how='any'); df1.fillna(value=5); pd.isna(df1);  frame[frame['col1'].notna()]
df.apply(np.cumsum)
# Histogramming
s.value_counts() # It is ordered
tips.groupby('sex').size()
tips = tips.loc[tips['tip'] <= 9] # Delete
tips.loc[tips['tip'] < 2, 'tip'] *= 2 #Update
# Top 10 rows with offset
tips.nlargest(10 + 5, columns='tip').tail(10)
# Top N rows per group
(tips.assign(rn=tips.sort_values(['total_bill'], ascending=False)
   ....:                     .groupby(['day'])
   ....:                     .cumcount() + 1)
   ....:      .query('rn < 3')
   ....:      .sort_values(['day', 'rn']))