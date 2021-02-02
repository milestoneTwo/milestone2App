# Milestone 2 Project

Brought to you by Dave Franks and Ermias Bizuwork [About](about.md)

MADS Program 2021 Go Blue!!

## About the Project

The project was broken up into two parts based on the type of learning being employed. The first part [Part A](part1.md) is the supervised learning portion where the goal was to build a binary classifier to beat some derived estimators at predicting the level of reading difficulty of text. The second part [Part B](part2.md) is the unsupervised learning portion of the project. Where using the same data the goal was to use unsupervised learning techniques to glean some information from the dataset. 

In summary there were two main purposes to the project.

- [Part A](part1.md): To build a model that out performs in accuracy metric the Logistic Regression Classifier and Naive Bayes Classifier scoring 68% and 65% accuracy respectively.
- [Part B](part2.md): Create an unsupervised learning model that can characterize and explain the dataset in some way.

The third and final stretch goal was to combine the two parts to have a semi-supervised algorithm to better increase the accuracy of the binary classifier made in Part A. The idea behind was to build a feature using an unsupervised topic model like [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to create a feature that would be aggregated with other features in Part A. Due to some technical limitations it may not be feasible to accomplish this portion of the task but a description of what could be done is described in the [Stretch Goals](stretchgoals.md)

## Using the Project

A good starting point which you have already achieved if you are seeing this message is going to the [github repo](https://github.com/milestoneTwo/milestone2App) and following the instructions in the README.md. This will help you get the project setup if you haven't already and get your development environment ready to go. 

## Installation Steps
#### To Setup Local Environment

1. Clone Directory <br>
`git clone https://github.com/milestoneTwo/milestone2App.git`
2. Change directory inside the directory <br>
`cd [project directory]`
3. Make virtual environment <br>
`python3 -m venv m2venv`
4. Jump inside the virtual environment<br>
`source m2venv/bin/activate`
5. Install requirements <br>
`pip install -r requirements.txt`

#### Run Steps

To run the development web server run the following commands

`cd Milestone2Docs && mkdocs serve`
Or
`python app.py --run app`

Build the app with source data
{**DO NOT DO THIS**}

`python app.py --run build_app`

## Project Design Notes
