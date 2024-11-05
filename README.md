# Female Labour Force Participation (FLFP) Prediction and Analysis

## Project Overview
This project explores the application of statistical and machine learning models to predict Female Labour Force Participation (FLFP) using a dataset from the World Bank Survey. The main objective is to assess the predictive performance of three different algorithms—Linear Regression, Decision Tree, and Random Forest—and identify the strengths and weaknesses of each in relation to FLFP.

## Table of Contents
1. [Declaration](#declaration)
2. [Certificate](#certificate)
3. [Acknowledgement](#acknowledgement)
4. [Abstract](#abstract)
5. [Introduction](#introduction)
6. [Methodologies](#methodologies)
   - [Linear Regression](#linear-regression)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
7. [Data](#data)
8. [Implementation](#implementation)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [References](#references)

## Abstract
Female Labour Force Participation (FLFP) is an essential topic in labour and behavioural economics. This project compares the effectiveness of three methods—Random Forest, Decision Tree, and Linear Regression—applied to FLFP data. The results provide insights into which model best handles the explanatory factors and offers the most accurate predictions.

## Introduction
Labour force participation and its determinants are core to economic studies. This project evaluates FLFP using predictive models, addressing various factors such as market conditions, family income, education, fertility rates, and marital status. The study sheds light on gender disparities in labour force participation and explores trends over recent decades.

## Methodologies

### Linear Regression
Linear Regression is a fundamental predictive model that assesses the relationship between a dependent variable and one or more independent variables. It is used here to understand FLFP and identify key predictive factors.

#### Why Linear Regression?
Linear Regression offers a simple yet effective approach to making predictions, widely applicable in business and social sciences due to its efficiency and reliability.

#### Assumptions for Effective Linear Regression
Several assumptions must be met for Linear Regression to be accurate, including data normality, independence, and linearity.

### Decision Tree
Decision Trees classify data by splitting it into subsets based on attribute values, forming a tree-like structure. The algorithm selects the most informative features at each step, making it intuitive and easy to interpret.

### Random Forest
Random Forest is an ensemble technique that combines multiple Decision Trees to improve prediction accuracy. It is effective for both regression and classification tasks and handles large datasets with high-dimensional spaces efficiently.

## Data
The dataset is sourced from the World Bank Survey and includes variables influencing FLFP, such as education, marital status, and other economic indicators.

## Implementation
The project is implemented using Python, primarily utilizing the following packages:
- **NumPy** for data manipulation
- **scikit-learn** for building and evaluating the models

The implementation steps include data preprocessing, model training, testing, and evaluation.

## Results
The Mean Squared Error (MSE) values for each model are as follows:
- **Linear Regression:** 24.93
- **Decision Tree:** 43.89
- **Random Forest:** 0.64

These results indicate that the Random Forest model outperforms both Linear Regression and Decision Tree models in terms of predictive accuracy.

## Conclusion
The study demonstrates that the Random Forest model provides superior predictive accuracy for FLFP. While Decision Trees and Linear Regression offer interpretability, Random Forest is preferred for its robustness and stability.

## References
1. [Estimating Female Labor Force Participation through Statistical and Machine Learning Methods](https://www.researchgate.net/publication/228998196_Estimating_Female_Labor_Force_Participation_through_Statistical_and_Machine_Learning_Methods_A_Comparison)
2. [Understanding Random Forest](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
3. [Introduction to Random Forest Algorithm](https://www.analyticsvidhya.com/blog/2021/10/an-introduction-to-random-forest-algorithm-for-beginners/)
4. [Decision Tree Classification](https://medium.com/@pranav3nov/decision-tree-classification-5916bba46b1a)
5. [Introduction to Linear Regression](https://www.ibm.com/in-en/topics/linear-regression)
6. [World Bank Dataset](https://data.worldbank.org/indicator/SL.TLF.CACT.FE.ZS)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
