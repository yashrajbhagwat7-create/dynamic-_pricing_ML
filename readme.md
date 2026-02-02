📈 Dynamic Pricing Recommendation System (Machine Learning)

⚠️ Educational Project
This project demonstrates how machine learning can be used to model demand elasticity and analyze revenue-optimal pricing strategies.
It is not intended for real-world pricing deployment.

📌 Project Overview

Pricing directly impacts demand and revenue.
This project builds a machine learning–based pricing recommendation workflow that:

Models the relationship between price and demand

Captures price elasticity

Simulates different pricing scenarios

Identifies revenue-optimal price points

The focus is on data generation, feature engineering, modeling, and evaluation, which are the core ML components behind real pricing systems.

🎯 Objectives

Generate realistic synthetic sales data with proper price elasticity

Engineer meaningful time-based and price-based features

Train and compare ML models for demand prediction

Simulate pricing scenarios and compute expected revenue

Recommend prices that maximize revenue under simulated conditions

🗂️ Project Structure
dynamic-pricing-ml/
│
├── data/
│   └── generate_sales_data.csv
│
├── eda/
│   └── eda_data_analysis.py
│
├── features/
│   └── feature_engineering.py      # Feature creation & preprocessing
│
├── models/
│   └── demand_model.py              # Model training & evaluation
│
├── pricing/
│   └── price_simulator.py           # Price–demand & revenue simulation
│
├── README.md
└── requirements.txt

🧪 Dataset

Synthetic dataset (365 days) with the following fields:

date

price

units_sold

promotion (binary)

Time features (day of week, month)

Seasonal patterns

Negative price elasticity (higher price → lower demand)

Why synthetic data?

Allows controlled elasticity behavior

Avoids noisy or biased real-world datasets

Keeps the focus on ML concepts, not data collection

🧠 Feature Engineering
Price-Based Features

price

price_squared

log(price)

Seasonality Features

Sine/Cosine encoding for day of week

Sine/Cosine encoding for month

Promotions

Binary promotion indicator

These features help the model learn smooth demand trends instead of rigid rules.

🤖 Models Used
Model	MAE	MSE
Linear Regression	~7.9	~100
Random Forest	~10.4	~167
Why Linear Regression performed better

Captures elasticity relationships more cleanly

Produces smoother price–demand curves

Easier to interpret for pricing analysis

💰 Pricing Simulation & Revenue Optimization

Pricing simulation was performed by:

Varying price over a defined range

Predicting demand for each price

Computing revenue:

revenue = price × predicted_demand


Selecting the price with maximum expected revenue

Example Output
Best Price: 80.0
Predicted Demand: 209.28
Expected Revenue: 16,742.56


This demonstrates how ML predictions can inform pricing decisions.

🏗️ Pricing Engine (Design Note)

In real-world systems, model predictions would flow through a business-constrained pricing engine, enforcing rules such as:

Maximum allowable price change

Minimum demand thresholds

Inventory constraints

Regulatory bounds

This project focuses on ML modeling and revenue analysis, which are prerequisites for such systems.

📌 Key Learnings

Why price elasticity is critical in pricing models

How poor synthetic data can break pricing logic

Importance of feature engineering for demand modeling

Why simpler models can outperform complex ones in pricing

Difference between prediction systems and decision systems

🚀 Future Improvements

Add inventory-aware pricing constraints

Introduce competitor pricing signals

Use probabilistic demand modeling

Deploy as an API-based pricing service

Add visualization dashboards

🧑‍💻 Author

Yashraj Bhagwat
Aspiring Data Scientist / Machine Learning Engineer
