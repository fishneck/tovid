# TOVID: Trend in COVID-19 Related Academic Literature


The massive size and the diversity of topics in thousands of COVID-19 related publications makes it an overwhelming task to manually detect the sub-categories, not to mention the change in research trend over time. Statistics-bases topic models are effective tools to analyze the semantic style in large document collections. However, incorporating temporal coherence and maintaining a fair interpretation of the results remain as challenging tasks, especially when the study object is highly real-life based. The aim of this study is to integrate a machine learning point of view and a comprehensive topic evolution framework into the recently flourishing field of SARS-COV-2 studies and propose a retrospective thinking on how the science community could better fight against the disease.


The respotory contains source codes for data cleaning(**preprocessing.py**) and topic modeling(**LDA.py** and **DTM.py**), as well as visualization results(**.ipynb**). The **LDA.py** contains both the training and evaluation while the **DTM.py** only plays the role of buliding the model where results will be analyzed in the **.ipynb** file.
