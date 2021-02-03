# Stretch Goals

## Semisupervised Learning using LDA Topic Modeling

Unfortunately this portion of the assignment went incomplete. In theory this is the steps that would have been taken to complete a semisupervised model. 

1. Train LDA model on training dataset
2. Acquire topics and assign a topic to each document one of top K topics
3. Train a classifier to learn topic to label outcomes. 

## Build A More Cohesive Project

Combining both parts of the project into a cohesive pipeline with coherent architecture will make it easier to perform a semisupervised algorithm as well as saving on runtime. Features could be made unanimously and stored appropriately. 

## Investigate Spacy Architecture and Pipelines

SpaCy has great tools for pipelines and could potentially speed up feature and model building. Since the code is written in C it is faster and can utilize GPU and Multithreading. 