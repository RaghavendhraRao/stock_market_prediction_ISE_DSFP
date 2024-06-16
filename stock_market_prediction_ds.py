# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:30:47 2024

@author: Raghavendhra
"""


'''
Installing packages

Install the ucimlrepo package
    pip install ucimlrepo
    

'''

from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
  
# # fetch dataset 
# istanbul_stock_exchange = fetch_ucirepo(id=247) 
ise_data = fetch_ucirepo(id=247) 


print(type(ise_data))


print("keys: ", ise_data.keys(), "\n")

data_info = ise_data.data['features']
print(type(data_info))

print("print first 5 rows..!")
print(data_info.head())

print("\n","Data Columns: ", data_info.columns)

plt.figure(figsize=(12, 6))
plt.plot(data_info['date'], data_info['SP'], marker='o', linestyle='-', color='b', label='ISE')
plt.xticks(rotation='vertical')
plt.xlabel('Date')
plt.ylabel('SP')
plt.title('ISE Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# print(istanbul_stock_exchange)