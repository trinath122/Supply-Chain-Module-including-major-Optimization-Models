
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

class NetworkDesignModel:
    def __init__(self, facilities, demand_points, transport_costs, fixed_costs, capacities, demands):
        self.facilities = facilities
        self.demand_points = demand_points
        self.transport_costs = transport_costs
        self.fixed_costs = fixed_costs
        self.capacities = capacities
        self.demands = demands

    def solve(self):
        model = pulp.LpProblem("Facility_Location", pulp.LpMinimize)

        open_facility = pulp.LpVariable.dicts("Open", self.facilities, cat='Binary')
        ship = pulp.LpVariable.dicts("Ship", [(f, d) for f in self.facilities for d in self.demand_points], lowBound=0)

        model += (
            pulp.lpSum(self.fixed_costs[f] * open_facility[f] for f in self.facilities) +
            pulp.lpSum(self.transport_costs.loc[f, d] * ship[(f, d)] for f in self.facilities for d in self.demand_points)
        )

        for d in self.demand_points:
            model += pulp.lpSum(ship[(f, d)] for f in self.facilities) == self.demands[d]

        for f in self.facilities:
            model += pulp.lpSum(ship[(f, d)] for d in self.demand_points) <= self.capacities[f] * open_facility[f]

        model.solve()

        result = {
            'Status': pulp.LpStatus[model.status],
            'OpenFacilities': {f: open_facility[f].varValue for f in self.facilities},
            'Shipments': {(f, d): ship[(f, d)].varValue for f in self.facilities for d in self.demand_points}
        }
        return result

class InventoryManagementModel:
    def __init__(self, sales_data, ordering_cost, holding_cost):
        self.sales_data = sales_data
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost

    def forecast_and_eoq(self, plot=False):
        self.sales_data['Month_Num'] = np.arange(1, len(self.sales_data) + 1)
        X = self.sales_data[['Month_Num']]
        y = self.sales_data['Sales']

        model = LinearRegression().fit(X, y)
        self.sales_data['Predicted'] = model.predict(X)

        future = pd.DataFrame({'Month_Num': [len(self.sales_data) + i for i in range(1, 4)]})
        future_sales = model.predict(future)

        mae = mean_absolute_error(y, self.sales_data['Predicted'])
        rmse = np.sqrt(mean_squared_error(y, self.sales_data['Predicted']))

        annual_demand = self.sales_data['Sales'].sum()
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / self.holding_cost)

        if plot:
            plt.figure(figsize=(10,5))
            plt.plot(self.sales_data['Month_Num'], self.sales_data['Sales'], label='Actual Sales', marker='o')
            plt.plot(self.sales_data['Month_Num'], self.sales_data['Predicted'], label='Predicted Sales', marker='x')
            plt.title('Sales Forecast vs Actual')
            plt.xlabel('Month Number')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.show()

        result = {
            'Forecast': future_sales,
            'EOQ': eoq,
            'MAE': mae,
            'RMSE': rmse,
            'PredictedSales': self.sales_data
        }
        return result
