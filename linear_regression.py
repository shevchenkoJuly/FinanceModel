import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
file_path = os.path.join(current_dir, r'C:\Users\Юля\Desktop\Python\Modeler_test_basic_ukr_v2.xlsx')

# Підготовка даних для аналізу:
data_macro = pd.read_excel(file_path, sheet_name='Input macro')
portf_migration_rate = pd.read_excel(file_path, sheet_name='Input_port')
portf_migration_rate['Portfolio forward migration rate'] = portf_migration_rate['Portfolio forward migration rate'].astype(str)
portf_migration_rate['Portfolio forward migration rate'] = portf_migration_rate['Portfolio forward migration rate'].str.replace(',', '.').str.rstrip('%')
portf_migration_rate['Portfolio forward migration rate'] = portf_migration_rate['Portfolio forward migration rate'].astype(float) 

# x - залажні та (y) не залежна змінна для моделі:
x = data_macro[(data_macro['date'] >= '2014-12-01') & (data_macro['date'] <= '2016-04-01')]
x = x[['date', 'CCPI_MY', 'Industr_M', 'Constr_MYC', 'Unemp_M', 'Wages_M']]
y = portf_migration_rate[(portf_migration_rate['date'] >= '2014-12-01') & (portf_migration_rate['date'] <= '2016-04-01')]['Portfolio forward migration rate']

# Розділення даних на тренувальний та тестувальний набори:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Побудова моделі лінійної регресії:
model = LinearRegression()
model.fit(x_train[['CCPI_MY', 'Industr_M', 'Constr_MYC', 'Unemp_M', 'Wages_M']], y_train)
y_pred = model.predict(x_test[['CCPI_MY', 'Industr_M', 'Constr_MYC', 'Unemp_M', 'Wages_M']])

# Оцінка моделі (вирахування похибки):
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Коефіцієнти:
coefficients = pd.DataFrame(model.coef_, columns=['Coefficient'], index=['CCPI_MY', 'Industr_M', 'Constr_MYC', 'Unemp_M', 'Wages_M'])
print('Coefficients:')
print(coefficients)

# Виведення константи (intercept) моделі:
intercept = model.intercept_
print(f'Intercept: {intercept}')

# Тестування моделі (розрахунок прогностичних данних):
for el in range(len(y_test)):
    actual_port = round((y_test.iloc[el] * 100), 2)
    forecast_port = round((y_pred[el] * 100), 2)
    print(f'Year: {x_test.iloc[el]["date"].year}, Portfolio forward migration rate: {actual_port}, Predicted: {forecast_port}')