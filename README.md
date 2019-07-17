# Humpback Whale Identification

### Training Data

Training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an Id. The challenge is to predict the whale Id of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.


### Approach

My Solution to the problem is to use Siamese Neural Networks to compute dissimilarity between one whale image with all 3000+ whale classes. After computing the dissimilarity of a whale image with all target categories we choose top 5 least dissimilar classes as top 5 predictions i.e. the whale classes which have least dissimilarity to current example in consideration is highly likely to fall into one of 5 categories.

As we have 25000+ images in training data with 5005 target classes the approach to compute dissimilarity with every image in training data is quite expensive computationally. To  reduce this sample size we use [Linear Assignment Problem](https://en.wikipedia.org/wiki/Assignment_problem) to figure out which whale image to use for representing a particular whale category. So finally to generate prediction for a whale image we compute dissimilarity with 5005 images (i.e. one whale image representing one whale category) and use these dissimilarity values to generate top 5 predictions.

### Evaluation Metric

Mean Average Precision @ 5 (MAP@5)


### Results

Our solution has an accuracy of 90.003%, you can find the kernel [here](https://www.kaggle.com/axel81/siamese-baseline-lb-0-822)
