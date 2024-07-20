# OilyGiant: Where to invest in wells?

## Contact
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andres946/)
[![Correo Electrónico](https://img.shields.io/badge/Correo%20Electrónico-andresgvelasquez8@gmail.com-red?style=for-the-badge&logo=mail.ru)](mailto:andresgvelasquez8@gmail.com) 

## Description of the project

You work for the mining company OilyGiant. The task is to find the best location for a new oil well.
Steps to choose the location:

- Collect parameters of oil wells in the selected region: oil quality and reserve volume.
- Build a model to predict reserve volume in new wells.
- Select oil wells with the highest estimated values.
- Choose the region with the highest total profit for the selected oil wells.

Data on crude oil samples from three regions are available. Parameters of each oil well in the region are already known. Create a model to help choose the region with the highest profit margin. Analyze potential benefits and risks using the bootstrapping technique.

## Data Description

Geological exploration data for the three regions are stored in files:

- `../datasets/geo_data_0.csv`.
- `../datasets/geo_data_1.csv`.
- `../datasets/geo_data_2.csv`.

The files contain the following columns:

- `id` — unique identifier of the oil well.
- `f0`, `f1`, `f2` — three features of the points (their specific meaning is not important, but the features themselves are significant).
- `product` — volume of reserves in the oil well (thousands of barrels).

## Virtual Environment Setup

To run this project, it is recommended to create a Python virtual environment. You can do this by executing the following commands in your terminal:

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
.\venv\Scripts\activate   # For Windows
```

Then, you need to install the dependencies:

```bash
pip install -r requirements.txt
```
Once you have set up the virtual environment and installed the dependencies, you can run the project.

## Metrics:

1. **RMSE (Root Mean Square Error):** Measures the average magnitude of the errors between predicted and actual values.

Comparison Table of Results:

| Region | Predicted Average Volume | Actual Average Volume | RMSE  |
| ------ | ------------------------ | --------------------- | ----- |
|   0    |          92.47           |         92.58         | 37.68 |
|   1    |          68.95           |         68.44         |  1.02 |
|   2    |          94.97           |         95.07         | 40.15 |

- For regions 0, 1, and 2: The overall average of predicted and actual reserves for the validation set is similar.
- Region 1 achieved significantly lower RMSE. This is because the dataset behaves uniformly except at the extremes. Wells tend to behave similarly in this region. However, the average is lower compared to other regions, being approximately 24% lower than others.

## Conclusions

The recommendation for the development of oil wells is Region 1 for the following reasons:

- The risk of loss is the lowest (2.1) compared to other regions.
- Region 1 achieved the highest average profit of $4,113,764.82 USD across 1000 samples.
- The 95% confidence interval has a positive minimum value compared to the other 2 regions ($174,946.08 USD).
- The 95% confidence interval has a range of $8,177,256.57 USD.


Comparison Table of Results:
| Region   | Average Profit    |  95% Confidence Minimum |  95% Confidence Maximum | Risk of Loss   |
|----------|-------------------|-------------------------|-------------------------|----------------|
| Region 0 |     3,789,115.39  |      -899,158.75        |       9,038,100.53      |       6.8      |
| Region 1 |     4,113,764.82  |        174,946.08       |       8,352,202.65      |       2.10     |
| Region 2 |     3,968,226.88  |     -1,403,306.60       |       8,773,711.43      |       7.6      |