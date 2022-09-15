# MLOps
MLOps is a methodology of operation that aims to facilitate the process of bringing an experimental Machine Learning model into production and maintaining it efficiently. MLOps focus on bringing the methodology of DevOps used in the software industry to the Machine Learning model lifecycle. In that way we can define some of the main features of a MLOPs project:
- Data and Model Versioning
- Feature Management and Storing
  - Preparing and maintaining high-quality data for training ML models 
  - Performing ongoing experimentation of new data sources, ML algorithms, and hyperparameters
- Automation of Pipelines and Processes
  - Maintaining the veracity of models by continuously retraining them on fresh data
- CI/CD for Machine Learning
- Continuous Monitoring of Models
  - Validating
  - Tracking experiments
  - Tracking models in production to detect performance degradation / gain
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Data and Model Versioning
The use of code versioning tools is vital in the software development industry. The possibility of replicating the same code base so that several people can work on the same project simultaneously is a great benefit. In addition, versioning these bases allows them to work in different sections in an organized manner and without compromising the integrity of the code in production.

As much as these tools solve several problems in software development, there are still issues in machine learning projects. Code versioning is still crucial, but when working on new experiments it's important to guarantee the same properties for data and models.

In a machine learning project, data scientists are continuously working on the development of new models. This process relies on trying different combinations of data, parameters, and algorithms. It's extremely positive to create an environment where it's possible to go back and forth on older or new experiments.
![image](https://user-images.githubusercontent.com/88195980/190283146-82f948b4-9cd2-4030-bd80-89bffbc49070.png)

### Reproducibility
When discussing versioning, it's important to understand the term reproducibility. While versioning data, models, and code we are able to create a nice environment for data scientists to achieve the ultimate goal that is a good working model, there is a huge gap between this positive experiment to operationalize it.

To guarantee that the experimentation of the data science team will become a model in the production for the project, it's important to make sure that key factors are documented and reusable. The following factors listed below were extracted from "Introducing MLOps" (Treveil and Dataiku Team 57) :

 - **Assumptions:** Data Scientist's decisions and assumptions must be explicit.
 - **Randomness:** Considering that some machine learning experiments contain pseudo-randomness, this needs to be in some kind of control so it can be reproduced. For example, using "seed".
 - **Data:** The same data of the experiment must be available.
 - **Settings:** Repeat and reproduce experiments with the same settings from the original.
 - **Implementation:** Especially with complex models, different implementations can have different results. This is important to keep in mind when debugging.
 - **Environment:** It's crucial to have the same runtime configurations among all data scientists.
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Feature Storing
### What is a Feature Store?
**Feature Stores** are components of data architecture that are becoming increasingly popular in the Machine Learning and MLOps environment. The goal of a Feature Store is to process data from various data sources at the same time and turn it into features, which will be consumed by the model training pipeline and the model serving. The concept of Feature Stores is novice and rapidly changing, therefore this page has the objective of showing the key features that are more common among the main Feature Stores in the market, but at the same time it is important to note that some of the tools and frameworks in the market might not comprehend all those exact characteristics in the same manner.
![image](https://user-images.githubusercontent.com/88195980/190283188-2704a8aa-57c1-412d-9d4d-aae40b99f841.png)
### Why it matters?
Feature Stores can be very useful for Machine Learning in production and are very reliable ways to manage features for research and training using Offline Stores, as it is to manage the feeding of features to a model served in production using an Online Store. This data component can manage to comprehend a wide spectrum of different projects and necessities, some of which are seen below.

**Key Features**
 - Enables features to be shared by multiple teams of Data Scientists working at the same time.
 - Creates a reliable automated preprocess pipeline of large quantities of data.
 - Can use and combine different data sources, such as data lakes, data warehouses and streaming of new data, all at once.
 - Provides relevant and online features to a model in production.
 - Can use a time windows system for Data Scientists to gather features from any point in time.
 - Highly customizable for different model needs of consumption, such as batch or real-time predictions.

### Offline Store vs Online Store
Feature Stores combine multiple data sources and preprocess those into features, the main types of data are:

 - **Batch Data:** Usually coming from Data Lakes or Data Warehouses. Those are big chunks of data that have been stored in order to be used by models and are not necessarily updated in real-time. Example: Data from customers of a bank, such as age, country, etc.

 - **Real-time Data:** Usually coming from Streaming and Log events. Those the online data that are constantly coming from sources like the events logged on a system. Example: A transaction in a bank is logged in real-time and fed to the Feature Store.

Those types of data are combined inside and form two types of stores:

 - **Offline Stores:** Store composed of preprocessed features of Batch Data, used for building a historical source of features, that can be used by Data Scientists in the Model Training pipeline. With it's historical components, in most Feature Stores it can be used to provide a series of features at a given time frame or time point. It is normally stored in data warehouses, like IBM Cloud Object Storage, Apache Hive or S3, or in databases, like PostgreSQL, Cassandra and MySQL, but it can also be used in other kinds of systems, like HDFS.
 - **Online Stores:** Store composed of data from the Offline Store combined with real-time preprocessed features from streaming data sources. It is built with the objective of being the most up-to-date collection of organized features, which can be used to feed the Model in Production with new features for prediction. It is normally stored in databases for rapid access, like MySQL, Cassandra, Redis, but it can be stored in more complex systems.
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Automation
### Why automate Machine Learning?
The automation of machine learning pipelines is highly correlated with the maturity of the project. There are several steps between developing a model and deploying it, and a good part of this process relies on experimentation.

Executing this workflow with less manual intervention as possible should result in:

- Fast deployments
- More reliable process
- Easier problem discovering

**Levels of automation**
Defining the level of automation has a crucial impact on the business behind the project. For example: data scientists spend a good amount of time searching for good features, and this time can cost too much in resources which can impact directly the business return of investment(ROI).

Lets describe a three-level of automation in machine learning projects:

- **Manual Process:** Full experimentation pipeline executed manually using Rapid Application Development(RAD) tools, like Jupyter Notebooks. Deployments are also executed manually.
- **Machine Learning automation:** Automation of the experimentation pipeline which includes data and model validation.
- **CI/CD pipelines:** Automatically build, test and deploy of ML models and ML training pipeline components, providing a fast and reliable deployment.

The next diagram describes process of model retraining

![image](https://user-images.githubusercontent.com/88195980/190283220-7d8316a4-028d-45f8-b7e3-0b78313224ee.png)
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## CI/CD for Machine Learning

Just like in DevOps, CI/CD is a method to make changes more frequently by automating the development stages. In machine learning(ML) this stages are different than a software development, a model depends not only on the code but also the data and hyperparameters, as well as deploying a model to production is more complex too.

![image](https://user-images.githubusercontent.com/88195980/190283262-1939ace3-136b-4c11-b54f-d82b69ca643f.png)

**Continuous Integration (CI)**

Continuous integration in ML means that every time a code or data is updated the ML pipeline reruns, this is done in a way that everything is versioned and reproducible, so it is possible to share the codebase across projects and teams. Every rerun may consist in training, testing or generating new reports, making easier to compare between other versions in production.

Note that, it is possible and recommended to run code tests too, for example, ensuring the code is in certain format, dataset values, such as NaN or wrong data types or functions outputs.

Some examples of a CI workflow:

- Running and versioning the training and evaluation for every commit to the repository.
- Running and comparing experiment runs for each Pull Request to a certain branch.
- Trigger a new run periodically.

**Continuous Deployment (CD)**

Continuous deployment is a method to automate the deployment of the new release to production, or any environment such as staging. This practice makes it easier to receive users' feedback, as the changes are faster and constant, as well as new data for retraining or new models.

Some examples of CD workflow:

- Verify the requirements on the infrastructure environment before deploying it.
- Test the model output based on a known input.
- Load testing and model latency.

The next diagram describes deployment process
![image](https://user-images.githubusercontent.com/88195980/190283285-099c2ffa-7ed1-42d8-90b5-b3047aae14a7.png)
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Continuous Monitoring
Machine Learning models are unique software entities as compared to traditional code and their performance can fluctuate over time due to changes in the data input into the model after deployment. So, once a model has been deployed, it needs to be monitored to assure that it performs as expected.

It is also necessary to emphasize the importance of monitoring models in production to avoid discriminatory behavior on the part of predictive models. This type of behavior occurs in such a way that an arbitrary group of people is privileged at the expense of others and is usually an unintended result of how the data is collected, selected and used to train the models.

Therefore, we need tools that can test and monitor models to ensure their best performance, in addition to mitigating regulatory, reputation and operational risks.

### What to Monitor?
The main concepts that should be monitored are the following:

- **Performance:** Being able to evaluate a model’s performance based on a group of metrics and logging its decision or outcome can help give directional insights or compared with historical data. These can be used to compare how well different models perform and therefore which one is the best.

- **Data Issues and Threats:** Modern models are increasingly driven by complex feature pipelines and automated workflows that involve dynamic data that undergo various transformations. With so many moving parts, it’s not unusual for data inconsistencies and errors to reduce model performance, over time, unnoticed. Models are also susceptible to attacks by many means such as injection of data.

- **Explainability:** The black-box nature of the models makes them especially difficult to understand and debug, especially in a production environment. Therefore, being able to explain a model’s decision is vital not only for its improvement but also for accountability reasons, especially in financial institutions.

- **Bias:** Since ML models capture relationships from training data, it’s likely that they propagate or amplify existing data bias or maybe even introduce new bias. Being able to detect and mitigate bias during the development process is difficult but necessary.

- **Drift:** The statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes, producing what is known as concept drift.

The following drawing shows that the health of a Machine Learning system relies on hidden characteristics that are not easy to monitor therefore using the analogy of an iceberg.

![image](https://user-images.githubusercontent.com/88195980/190283315-9adcaebf-514d-4bba-b28e-ec1f325c30e1.png)
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Big Two Comparison
| Name of Service | Additional Info |<IMG  src="https://nub8.net/wp-content/uploads/2019/07/amazon_sagemaker-min.png"  alt="Machine Learning with Amazon SageMaker » Nub8" width="250" height="75" >|<IMG  src="https://miro.medium.com/max/1127/1*-ganvHfXEbn6oYk-krRpIg.jpeg"  alt="Azure Machine Learning Service: Part 1 — An Introduction ..." width="150" height="75"> |
|--|--|:----:|:----:|
|Notebook Support||:heavy_check_mark:|:heavy_check_mark:|
|Jupyter Lab|Azure has entire custom interace|:heavy_check_mark:|:x:|
|Computer instance|Azure gives more control over compute|at set-up time|can change via notebook|
|Workflow pipeline support||:heavy_check_mark:|:heavy_check_mark:|
|Studio/Low code/GUI and Drag and Drop support||:heavy_check_mark:|:heavy_check_mark:|
|In-build Feature Store||:heavy_check_mark:|:x::heavy_exclamation_mark:Databricks FS can integrated:heavy_exclamation_mark:|
|Automatic ML||:heavy_check_mark: Auto pilot|:heavy_check_mark: Automated ML|
|Label the data|Both supporting outsorcing|:heavy_check_mark: Ground Truth|:heavy_check_mark:Data Labeling|
|GPU Support||:heavy_check_mark: Framework optimised|:heavy_check_mark:Can install GPU drivers on compute|
|Real time endpoint||:heavy_check_mark: Internal only via SDK|:heavy_check_mark: Internal via SDK & Public|
|Offline job||:heavy_check_mark: Batch Transform|:heavy_check_mark: Pipeline|
|In-build support for IoT||:heavy_check_mark: Neo|:x::heavy_exclamation_mark:Possible via IoT edge:heavy_exclamation_mark:|
|In between data visualisation||:x: Almost No or very basic|:heavy_check_mark:|
|A/B testing support||:heavy_check_mark:Traffic Routing|:x:|
|Reinforcement Learning||:heavy_check_mark:|:heavy_check_mark:|
|Kubernetes Suport||:heavy_check_mark:|:x:|
|Multiple Model on same endpoint to save cost||:heavy_check_mark:MMS|:x:|
|Automatic model debugging, tuning||:heavy_check_mark:Debugger Model Tuning|:x:|
|Auto Scaling, edge optimization of endpoint||:heavy_check_mark:|:heavy_check_mark:|
|Model monitoring||:heavy_check_mark:|:heavy_check_mark:|
|Responsible AI||:heavy_check_mark:Model Explainability|:heavy_check_mark: Model interpretability|
|Augmented AI||:heavy_check_mark:|:x:|
