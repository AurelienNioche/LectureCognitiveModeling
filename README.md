# LectureCognitiveModeling
Material for the class

Main claims: 

* Modeling is comparing!
* Simulating before experimenting is helping!

## 1. Task definition & Model definition

### 1.1 Task definition

#### Goal

- Check the adequacy with the question you want to reply to
- Select adequate characteristics (i.e., enough time-steps) 

### 1.2 ModelS definition

#### Goal

- Select/Define which aspect(s) you want to model
- Model alternatives

## 2. First simulations [RW only]

##### General Goals

- Have a general image of what your results could look like

### 2.0 Represent effect of free parameters on behavior

#### Goals

- Isolate the effect of each parameter
- Notice dynamics that may differ from initial intuitions (e.g., non linear)

#### Figures

* Learning rate: value over time for different values
* Temperature: probability of choice over difference of values

### 2.1 Simulate a single agent [RW only]

#### Goals

- Have a first insight into the expected behavior
- Be sure that your metrics are adapted

#### Figures

* Plot: purely descriptive
* Plot: maximizing readability

### 2.2 Analyse latent variables

#### Goals

- Observe dynamics of your model

#### Figures

* Q-values / p-choices + behavior over time IND

### 2.3 Simulate a population of agent

#### Goals

- Evaluate the 'noise' of your behavior by using constant parametrisation
- Get a picture of your expected behavior under the best scenario 
(you find the best model, and subjects share the same best parameters)

#### Figures

* Plot: latent variables + behavior over time POP


## 3. Parameter recovery [RW only]

#### General goals

- Be sure that you are able to retrieve the parameters of your model, 
assuming that your model is correct

### 3.1 In a single agent

#### Goals

- Be sure that in the simplest case (single subject), 
you retrieve close parameters

#### Figures

* Plot: Initial results + Best fit same hist + Best fit new history IND 

### 3.2 In an homogeneous population

#### Goals

- Generalize what you observe with a single subject to a population, 
maintaining constant the parameter set 

#### Figures

* Plot: Initial results POP + Best fit POP

### 3.3 Exploration of objective function values over parameter space

#### Goals

- Observe the behavior of your objective function over the parameter space
- Notify the local minima 

#### Figures

* Plot: objective function values over parameter space (phase diagram)

### 3.4 Statistical approach

#### Goals

- Test parameter recovery for a large (enough) set of parameters
- Have a metric of the quality of your parameter recovery (Pearson's r),
including statistical assessment of the relevancy

#### Figures

* Plot: scatter plot

## 4. Model comparison

#### Goals

- Ensure that each model, when used for simulating, is selected as the 
best model
- Have a metric for the quality of your 'model' recovery

#### Figures

* Confusion matrix

## 5. Fake experiment

#### Goals

- Have an overview about what the results of your experiment could look like,
under the assumption that you get the best model, 
but that there is small variations in the population 
in terms of parametrization

- Ensure that the metrics of your behavior is adapted

- Be sure that you can retrieve the model in this context

- Have an idea about the distribution of your metrics 
regarding model selection (log-likelihood sums, BIC)


#### Figures

* Fake experiment

