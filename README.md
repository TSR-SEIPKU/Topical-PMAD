# Topical-PMAD

This repository contains the codes and data for Topical PMAD Method

## Content of Files
File|Content
------------ | -------------
syn.py | Python script for generating synthetic data
exp-data.zip | The synthetic data we used in our experiments
others | Java maven project for Topical PMAD method, including implementations of baseline methods used in the experiments

## PMAD Java Project

The topical PMAD method is a Java maven project. The main algorithms are implemented based on mallet (http://mallet.cs.umass.edu). See org.pmad.TopicalPMAD and org.pmad.Experiments for details.<br>
**Important: Add the jars in the lib directory as dependencies to the project's running environment.**

## Data Format

A dataset is separately recorded in two files, each containing one type of feature for an instance in a line, in the following format:

```
INSTANCE_ID	FEATURE_TYPE	f1 f2 f3 f4 ... fn
```

INSTANCE_ID is the unique instance id. FEATURE_TYPE is the type of features in this file. The rest are the observed features. In our synthetic datasets, the IDs of normal instances are integers, and the IDs of anomalous instances start with "ab-".

