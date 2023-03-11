# 310-mleng-challenge
 
**The Answers to the 310.ai ML Engineering challenges [here](https://equatorial-sternum-35b.notion.site/ML-Engineer-22c6afe2d1f34c2b9b4b6f99d48da81b)**

----

## Intermediate **Task 1**


**Task Specification:**
1. Download and prepare data for [MIT Indoor Classification](https://web.mit.edu/torralba/www/indoor.html). The final form of your submission will be a Jupyter Notebook. We suggest that you use a [Google Colab](https://colab.research.google.com/). Also, we suggest that you use [PyTorch](https://pytorch.org/) to prepare your ML pipeline. When you are done with it, download your notebook and submit it to us at people@310.ai
2. Create a minimal baseline for the classification task using 20% of data as validation. Use [balanced-accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) as your metric. Name this notebook “Baseline”. In this part, we focus on your code being clean, simple and understandable. We expect a clear pipeline, going over data loading, model creation, training and evaluation.
3. Use any tricks you like (ex. model architecture, augmentation, etc.) to improve the metric. The only method you are not allowed to use is pretraining using other datasets (ex. ImageNet pretraining). A minimum balanced-accuracy of 55% is required. Your submission will be examined based on its performance on a random validation (not your own split). The focus on this part is to measure your ability to do critical thinking and come up with good ideas to attack a hard problem. Name this notebook “Challenge” in your submission.


### Solution
The main code's notebook is located at `/task1-IndoorScene/Challenge/Challenge.ipynb`. The trained models PyTorch objects are saved in the corresponding
directory, i.e. `/round3checkpoints` and could be used for inference on new unseen dataset via the last cell of the notebook. Results and conclusions are all provided in the notebook.

Also, the baseline model's code is located at `/task1-IndoorScene/Baseline/Baseline.ipynb` along with the explanations and instructions.

---

## Intermediate **Task 2**

**Task Specification:**
1. Read this tutorial [https://pytorch.org/tutorials/beginner/transformer_tutorial.html](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
2. Change the task from autoregressive to predict the masked words. You can mask 10% of the words at random.
3. Define a reasonable baseline and compare model performance with that.
4. Try to improve the model performance without using more resources.
5. Improve the model without any restrictions. You can update architecture, hyper parameters etc.


### Solution
The second task's solution notebook is located at `/task2-MLM/MLM_challenge.ipynb`. The step-by-step walkthrough of the code is included in the notebook along with the the Results and conclusions are all provided in the notebook.

---

## **Task 3 [**TODO**]**



1. Read [https://github.com/Shen-Lab/TALE](https://github.com/Shen-Lab/TALE) 
2. Move the code to Pytorch 
3. Improve the performance with minor changes to the architecture and without using more resources
4. Improve the performance by designing a totally different architecture. You can design your own auxiliary tasks and even bring additional datasets.
