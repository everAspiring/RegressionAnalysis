
                                     Predicting Restaurant Profits: Regression Analysis
Introduction
Regression in Machine Learning is a Supervised Learning mechanism which uses two datasets. a) Training dataset and b) Testing dataset. 
Training dataset consists of input data and known response values. Using training dataset, the algorithm builds a model to predict the response values of the testing dataset. Testing dataset (previously unseen data) is used to validate the model. Often, larger training datasets yield more accurate results.
Problem Statement
Suppose a restaurant already has branches in different cities and it maintains data of profits and populations from all its outlets. Based on this data, it is considering various cities to open new outlets. This study aims to estimate the profits in new cities using population-profit data of outlets in existing cities. Thus, it helps the restaurant to select a city to expand its business. The relationship between population and profit is best studied through Regression. The idea here is implemented in Python language.
Technologies used: 
<table align="center">
<tr><td>Language</td>	<td>Python 3.5</td></tr>
<tr><td>IDE</td>	<td>Anaconda</td></tr>
<tr><td>OS</td>		<td>Linux</td></tr>
<tr><td>Server</td>	<td>IPython Notebook Server</td></tr>
</table>

What is regression analysis? <br />
The term “regression” in scientific terminology refers to “prediction”. Regression Analysis in Machine Learning is a predictive modeling technique that analyses the relationship between two or more variables [1]. It is of two types.
1) Linear Regression
   2) Non-Linear or Logistic regression
Logistic regression deals with classifying data into two or more classes. The Linear Regression analyses the relationship between two (Simple Linear Regression) or more variables (Multiple Linear Regression). It uses one or more variables (“population”) and estimates the value of another variable (“profit”). A “linear” relationship is observed since the value of the dependent variable (“profit”) is directly proportional to the value of independent variable (“population”). In Regression, the dependent variable is always continuous and independent variables can be discrete or continuous. 
Why Simple Linear Regression?
Simple Linear Regression (SLR) is a widely used modeling technique. SLR estimates the relationship between two variables. The dataset used in this study includes two variables namely population and profit [2]. The variables share a linear relationship in the sense that as the population increases (or decreases), the sales increase (or decrease) and hence the profits increase (or decreases).
How does SLR work?
The SLR builds a linear function to predict the value of dependent (target) variable using independent variables (input). Mathematically, this function can be expressed in slope-intercept (line) equation.
				         y=mx+c                           
where, y=dependent variable
m=slope
x=independent variable
c=intercept 
Plotting the data
The training and testing datasets used in this study are already preprocessed and hence free of duplicate or missing attribute values. The datasets consist of population attribute (continuous and measured in 10,000s of people) and a profit attribute (continuous and measured in $10,000). A few entries from the training dataset (total size = 98) are shown below. 

<p align="center">
  <img src="/SampleData.png" width="150"/>
</p>

 
The Python code snippet used to implement is,
<p align="left">
  <img src="/Code1.png" width="250"/>
</p>
 
Data visualization using a Two-Dimensional Scatter Plot:
 <p align="center">
  <img src="/PlottedGraph_Outlier.png" width="450"/>
</p>
The data in x-axis represents the Population of City in 10,000s and y-axis represents the Profit made by the restaurant in $10,000s. The values in y-axis are the result of mapping linear functionality of values of x-axis. Hence it can be observed that as the population raises, the profits are raising in linear fashion.
Presence of an Outlier
An outlier is a data point that is different than many other data points. The data point circled in red (in the above plot) is detected as an outlier. 
Eliminating the Outlier
This outlier was dropped as it contradicts the assumption [3] that in a place with less population, minimum profits are expected. Although economy rate of the small town could affect the profits, this outlier is eliminated considering the attributes available in the dataset. The scatter plot after dropping the outlier is shown below.
 
After eliminating one outlier, the size of the training dataset is reduced to 97.
Regression model fit
The next step is to fit a regression line (model) that represents the hypothesis of the relationship of the population of the city and the profits that are to be predicted. Python’s Numpy library provides “polyfit()” function that essentially gives the intercept and slope values of the straight line represented as theta[0] and theta[1].
 
Is this the “best-fit” line?
With varying slope and intercept values, multiple regression lines could possibly fit the given data as shown below.
   
At this stage, it is important to determine the “best-fit” model (line) that fits the data. This can be accomplished by the Cost/loss function.
Cost function
The measure of difference between the estimated (hypothesis(x)) and actual result (y) is nothing but cost or loss/error. The goal is to minimize this error. The cost function essentially uses the “Sum of the squared differences” or “Least Square Method”. The Cost function is commonly used way to fit the regression line. It can be expressed in mathematical terms as follows.
				〖Cost/error=(hypothesis(x)-y)〗^2
where, hypothesis(x)=mx+c  = the estimated output 
with slope m, intercept c and a size of n.
y = the actual output
The above equation represents the cost of only one data point. Since overall cost is to be calculated, which is a measure of goodness of the regression line itself, the average cost values of all values are to be squared and added. Mathematically represented as below. 			
Cost=1/2n ∑_(i=1)^n▒(hypothesis(x_i)- y_i))^2 
The cost function when applied to the given dataset is illustrated as follows.
 
The least squares method calculates the vertical distance from each data point to the regression line and then squared and added so that there is no need to cancel out the positive and negative figures.
For example, below is the cost function value with slope value 1 and intercept value 2 for the given data.
 
How to choose slope and intercept parameters?
Now the goal is to minimize the cost function as much as possible. This means the difference between estimated and actual values is to be minimized. As the slope and intercept values of the line varies, the regression line changes accordingly. The values of slope and intercept are to be chosen such that the line fits most of the data points.
Gradient Descent: Optimization algorithm
One way to minimize the cost function is by using batch Gradient Descent optimization algorithm. This algorithm updates the slope and intercept values with each iteration by using the derivative. 
θ_j ∶=θ_j-α  ∂/(∂θ_j )  Cost(θ_0,θ_1)
A derivative is nothing but the slope of the line tangent to a data point. The slope at a data point gives the rate of change of cost function. The sign of the derivative indicates the direction to go to reduce the cost function. The number of steps to be taken is indicated by the learning rate α (alpha). The value of alpha should not be too large or too small. At every data point, the algorithm adjusts the parameter values such that at each point optimized values of slope and intercept are calculated.
Steps in Gradient Descent:
Step 1: Initialize θ_0 and θ_1 to some values (intercept and slope)
Step 2: Iteratively update θ_0 and θ_1 in ways that always reduces cost function
Deriving the partial derivative of cost function results in the following equations for the two parameters.
 θ_0 ∶=θ_0-α □(1/m) ∑_(i=1)^m▒(h_θ (x^(i ))- y^i  )) 
θ_1 ∶=θ_1-α □(1/m) ∑_(i=1)^m▒(h_θ (x^(i ) )- y^i  )〖(x〗_j)) 
Where θ_j = {θ_0, θ_1} = Intercept and Slope respectively
α = Learning Rate = 0.01
:=  assignment operator
m = Size of the dataset = 97
h_θ (x) = Hypothesized or observed value (profit) given x (population) as input
y = Actual value of profit
One way to verify if Gradient Descent is performing optimization correctly is by observing the Cost function values. The values of the Cost function should never increase. Rather should steadily decrease with each iteration and converge at some optimized value after all iterations are done with. With {0,0} as initial values of θ_j, learning rate as a constant value of 0.01 and number of iterations as 1500, the gradient descent function is invoked.
 
The values of gradient, theta and the history of cost function is at each iteration is printed (screenshot below).  
Interpreting the results
It can be observed that cost history has never increased but rather decreased by each passing iteration and reached a global optimum of value 4.4834. 
 
The values of theta are updated with initial values of {0,0} and calculated optimal values of {-3.62, 1.16}. 
With the updated (a.k.a optimized) θ_j  values obtained from executing gradient descent, the new model is replotted as below. 
The red marks represent the training data samples, and the blue line represents the hypothesized values of testing dataset. 
 
Summary of the results
 
Understanding the table values
Df residuals and model refers to degrees of freedom – number of values free to vary. The coefficient of 1.1930 means that as variable x is increased by 1, y will be increased by 1.1930. R-squared represents the percentage of variance. And std error represents the standard error (0.719). The interval [0.025 0.975] represents the 95% confidence interval that y value is between 0.025 to 0.975.
Visualizing using Surface and Contour graphs
The left subplot, Surface plot, represents the 3D view of the cost function with varying theta values. The right subplot, Contour plot, shows the minimum theta values calculated using GD. The final hypothesis equation returned by GD is as follows.
y = theta [0] + (theta [1]).x
y = -3.62981201+ (1.16631419) x 
The optimal values of intercept and slope returned by GD are -3.62981201 and 1.16631419 respectively. 
Verifying using Normal Equation
The theta values can be compared against a normal equation solution. The results obtained:
 
It can be observed that the minimum values obtained above is close to the values predicted by the model confirming the prediction accuracy.

References
[1] Adi Bronshtein. (2016, May, 18). Simple and Multiple Linear Regression in Python (1st ed.). [Online]. Available: https://medium.com/towards-data-science/simple-and-multiple-linear-regression-in-python-c928425168f9
[2] Dataset: https://www.kaggle.com/lalitsomnathe/population-profit
[3] Karen. (2016, July). Outliers: To drop or Not to Drop. [Online]. Available: http://www.theanalysisfactor.com/outliers-to-drop-or-not-to-drop/
