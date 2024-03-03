# DiabGuard: AI for Proactive Diabetes Prevention with Autoformer Transformer Analysis

![Code review](https://github.com/IEEE-NOVA-SB/DiabGuard-Proactive-Diabetes-Prevention-with-Autoformer-Time-Series-Analysis/blob/main/pair_programming.png
)

DiabGuard leverages advanced time series analysis with the Autoformer deep learning transformer to predict and prevent diabetes onset. 

By analyzing historical data, it provides personalized insights, empowering at-risk individuals with proactive health measures.

[Autoformer research paper](https://arxiv.org/abs/2106.13008)

[Autoformer hugging face link](https://huggingface.co/docs/transformers/main/model_doc/autoformer)


## Project Description

### What your application does?

DiabGuard, uses advanced time series analysis techniques powered by Autoformer, an AI model, to predict and prevent the onset of diabetes. 

By analyzing historical data and identifying patterns, DiabGuard offers personalized insights and recommendations for individuals at risk, empowering them to take proactive measures towards better health and diabetes prevention. 
 
### Why you used the technologies you used?

We chose to use Autoformer, an advanced AI model, for its ability to **effectively analyze time series data** and make accurate predictions. 

Additionally, Autoformer integrates seamlessly with PyTorch and transformers, providing efficient implementation of our predictive model, using the Hugging face transformer library.

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) was selected for its streamlined training process, enabling faster development and deployment of our solution.    
    
### Some of the challenges you faced and features you hope to implement in the future?

One challenge we faced was ensuring the scalability and efficiency of our model, especially when dealing with large volumes of time series data. 

In the future, we aim to enhance the interpretability of our predictions and further optimize the performance of DiabGuard. 

Additionally, we plan to incorporate real-time monitoring capabilities and expand the scope of our application to include additional risk factors for diabetes.



# Table of Contents
### [ How to Install and Run the Project ](#How_to_install)

### [ How to Use the Project ](#How_to_use)

### [ How to Contribute to the Project ](#how_to_contribute)

### [ Include Credits, Authors and acknowledgment for contributions ](#credits)


----



<a name="How_to_install">

# How to Install and Run the Project

### 1. Create a Virtual Environment (if not already created):
If you haven't already created a virtual environment for your project, you can do so using virtualenv or venv. Here's an example using venv:

```
python -m venv myenv
```


Replace ```myenv``` with the desired name for your virtual environment.

### 2. Activate the Virtual Environment:
On Windows, activate the virtual environment using:

```
myenv\Scripts\activate
```


On macOS and Linux, use:
```
source myenv/bin/activate
```
Replace ```myenv``` with the name of your virtual environment.


### 3. Install dependencies:
Once the virtual environment is activated, you can install Jupyter Notebook using pip:

```
pip install jupyter ipykernel torch lightning transformers ucimlrepo
```
This will install Jupyter Notebook within your virtual environment.

### 4. Verify Jupyter Installation:
To verify that Jupyter Notebook is installed in your virtual environment, you can run:


```
jupyter --version
```

This should display the version of Jupyter Notebook installed within your virtual environment.

### 5. Create a Jupyter Notebook Kernel for the Virtual Environment:
You need to create a Jupyter Notebook kernel that is associated with your virtual environment. This allows you to use the packages installed in your virtual environment within Jupyter Notebook.

#### a. First, activate your virtual environment (if it's not already activated).

#### b. Install the ipykernel package within the virtual environment:

```
pip install ipykernel
```
#### c. Now, you can create a Jupyter Notebook kernel for your virtual environment:


```
python -m ipykernel install --user --name=myenv --display-name="name"
```

Replace ```myenv``` with the name of your virtual environment and choose a suitable display name.

### 6. Start Jupyter Notebook:
Now, you can start Jupyter Notebook from within your virtual environment:

```
jupyter notebook
```
This will open a new Jupyter Notebook session in your web browser, and you should be able to select the "My Virtual Environment" kernel when creating a new notebook. This kernel will use the packages installed in your virtual environment.
</a>

<a name="How_to_use">


#### How to Use the Project

Run all jupyter notebooks cells
</a>


<a name="how_to_contribute">


#### How to Contribute to the Project

Make a pull request

</a>

<a name="credits">

#### Include Credits, Authors and acknowledgment for contributions

</a>

[Tiago Monteiro](https://www.linkedin.com/in/tiago-monteiro-)
