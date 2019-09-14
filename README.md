# Text-Classification-for-text-articles
Train multiple linear SVM classifiers to predict binary classes and classify into 4 classes


#### Classify each text article into one of the predfined classes/categories.
Here we have four sets of articles about: 1) operating systems, 2) vehicles, 3) sports, and 4)
politics. Each set consists of various articles collected from a corresponding newsgroup where
each article is represented by Bag-of-Words (BoW) features. That is, after extracting all the
relevant words from the entire dataset, feature vectors are dfined as the frequency of words
occurring in each article.

Since you have more than two classes, using a single binary SVM classifier is not sufficient.
Instead, you are going to combine several binary SVM classifiers to predict multiple classes.
Thus, the goal is to train multiple linear SVM classifiers that predict binary classes based
on BoW features, then combining them to properly predict one of the four classes. 

#### Data:	You can download the following data files in the class webpage.
	articles.train --> Training data consisting of 4,000 articles (1,000 articles per class)
	articles.test --> Test data consisting of 2,400 articles (600 articles per class)
	words.map --> A word table mapping every relevant word into its line number.
	
### Goal:

	1)	Train four dfferent (hard-margin) linear classiffers. As SVM classiffes only binary
		labels, you have to replace the target class number to 1 and all others to -1 before calling
		the library function. For instance, if you try to classify whether or not politics, you are to
		use 1,000 articles about politics as positive samples and 3,000 others as negative samples
		for training. Once you learn the four classiffers, the output label of each test example x.
	2)	Train soft-margin linear classiffers with different C values from {0.125,0.25, 0.5, 1, 2, 4, 8, ... , 256, 512}.
		In order to pick the best C value, you are required to perform a hold-out validation:
			1. Split the entire training data randomly into 75% for training and 25% for validation.
			2. For each C value, learn four binary classiffers similar to part (a) but only on the training data.
			3. Measure the overall classiffcation error on the validation data.
			4. Pick the C with the lowest validation error.
		Plot a graph showing both training and validation errors together with varying C in log-scale. 
		What are the best C value formulticlass classiffcation?

#### Results:

	Performed cross validation on 75-25 split.
	Using soft margin classifiers, we penalized few misclassifications with a loss function. 
	The C parameter is this loss function's multiplier; so the higher the C, the more we penalize making errors.
	C is a regularization parameter that controls the trade off between the achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data. When the values of C is too large the optimization algorithm tries to create a hyperplane which tries to classify each training example correctly.
	The best C value for multi-class classification is 20, after running for few values of C.
  
  #### To execute the code, download the R file and run it on R Studio.
