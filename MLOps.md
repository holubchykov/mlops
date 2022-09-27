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

General MLOps life cycle looks like this

![image](https://user-images.githubusercontent.com/88195980/191032396-3584ee05-54e9-44bf-b550-63d3d507e59c.png)


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
## Big Comparison

### Amazon SageMaker
Amazon SageMaker is a cloud machine-learning platform that helps users in building, training, tuning and deploying machine learning models in a production ready hosted environment.
Some Benefits of Using AWS SageMaker:
- Highly Scalable
- Fast Training
- Maintains Uptime — Process keeps on running without any stoppage.
- High Data Security

<IMG  src="https://d1.awsstatic.com/sagemaker/2022whiteupdate/overview-page/SageMaker%20Features.cafa389a0fa0e1c51230d9a4b68b4cce90e2dbfe.png"/>

This is how AWS pipeline looks like

![image](https://user-images.githubusercontent.com/88195980/192415507-7577f9e6-7cbb-4ea1-bae0-1b086212ac32.png)



### Azure ML

Azure Machine Learning Studio is Microsoft’s central point of contact for Machine Learning Computation in the Azure cloud. It is the successor of the Microsoft Machine Learning Studio Classic, which will be retired in 2024. Over the years Microsoft expanded the number of features and possibilities within their Azure ML Studio and the progress is still ongoing. Microsoft tries to make the creation of and the work with algorithms and experiments as simple as possible. Hence, you can expect that also Azure ML will get more and more features with automation and click surface than with hard-coded algorithms. But for now enough, let’s see the main components of the studio:

The Azure ML Studio menu is divided into three main sections: 
- Author. The section Author deals with the creation of the code and set-up of your machine learning processes.
  - Notebooks: This section encompasses everything around the creation, editing, and running of notebooks.
  - Automated ML: This section is for the easy automation and comparison of diverse machine learning techniques. You can easily input a dataset and see the performance of various ml techniques under diverse performance measures.
  - Designer: The Designer is a drag and drop tool to create your ML workflow from the input of a dataset to the evaluation. It is easy to use and has many features which can be searched via text or found by clicking on the arrows and captions on the left.
- Assets. Assets are the resources that are created and stored within the Author section such as the pipelines created in the Designer. It controls the resources’ whole workflow from inputting datasets over a pipeline to the endpoints for the output (e.g. connection to real systems via REST API).
  - Datasets: As the name states the section is used to register and manage your datasets.
  - Experiments: Experiments offer the possibility to measure runs with varying settings (e.g. hyperparamaters) and control the metrics.
  - Pipelines: Via the Designer, the pipelines can be created and are registered within the section pipeline to control the number of runs, runtime, name, etc.
  - Models: Useful to register and manage your models.
  - Endpoints: Here you can control and see all endpoints over which you can let other systems and applications use your models/algorithms
- Manage. The Manage section is for the system behind the scenes. It encompasses the computation clusters and instances, datastores on which the datasets are stored, and the integrations into other systems. It is somehow the invisible layer.
  - Compute: This section lets you set up, start and stop compute clusters and instances as well as inference clusters on which your models are running
  - Datastores: Comprise the storage on which datasets, models, notebooks, and all your files and data can be stored.
  - Data Labeling: Honestly, I do not understand why this section is under Manage and not under Author. I believe the Data Labeling would fit better to the first section since it has to do with the data handling, but maybe it refers to the “management of your data” and that’s why it is under Manage. Under the section, you get great possibilities and support to label all of your data examples.
  - Linked Services: This is for the integrations of other services and systems.

### Azure Databricks
Databricks Machine Learning is an integrated end-to-end machine learning platform incorporating managed services for experiment tracking, model training, feature development and management, and feature and model serving. The diagram shows how the capabilities of Databricks map to the steps of the model development and deployment process.

<IMG  src="https://learn.microsoft.com/en-us/azure/databricks/scenarios/media/what-is-azure-databricks/ml-diagram.png"  alt="What is Databricks Machine Learning?"/>


With Databricks Machine Learning, you can:

- Train models either manually or with AutoML.
- Track training parameters and models using experiments with MLflow tracking.
- Create feature tables and access them for model training and inference.
- Share, manage, and serve models using Model Registry.

For machine learning applications, Databricks provides Databricks Runtime for Machine Learning, a variation of Databricks Runtime that includes many popular machine learning libraries.

**Databricks Machine Learning features:**

- **Feature store**

Feature Store enables you to catalog ML features and make them available for training and serving, increasing reuse. With a data-lineage–based feature search that leverages automatically-logged data sources, you can make features available for training and serving with simplified model deployment that doesn’t require changes to the client application.

- **Experiments**

MLflow experiments let you visualize, search for, and compare runs, as well as download run artifacts and metadata for analysis in other tools. The Experiments page gives you quick access to MLflow experiments across your organization. You can track machine learning model development by logging to these experiments from Azure Databricks notebooks and jobs.

- **Models**

Azure Databricks provides a hosted version of MLflow Model Registry to help you to manage the full lifecycle of MLflow Models. Model Registry provides chronological model lineage (which MLflow experiment and run produced the model at a given time), model versioning, stage transitions (for example, from staging to production or archived), and email notifications of model events. You can also create and view model descriptions and leave comments.

- **AutoML**

AutoML enables you to automatically generate machine learning models from data and accelerate the path to production. It prepares the dataset for model training and then performs and records a set of trials, creating, tuning, and evaluating multiple models. It displays the results and provides a Python notebook with the source code for each trial run so you can review, reproduce, and modify the code. AutoML also calculates summary statistics on your dataset and saves this information in a notebook that you can review later.

- **Databricks Runtime for Machine Learning**

Databricks Runtime for Machine Learning (Databricks Runtime ML) automates the creation of a cluster optimized for machine learning. Databricks Runtime ML clusters include the most popular machine learning libraries, such as TensorFlow, PyTorch, Keras, and XGBoost, and also include libraries required for distributed training such as Horovod. Using Databricks Runtime ML speeds up cluster creation and ensures that the installed library versions are compatible.

This is how Azure Databricks looks like

![image](https://user-images.githubusercontent.com/88195980/192415825-55c95c56-624d-4241-ac4a-d2aa781d5b6c.png)

### Kuberflow
Kubeflow is a free and open-source machine learning platform designed to enable using machine learning pipelines to orchestrate complicated workflows running on Kubernetes. Kubeflow was based on Google’s internal method to deploy TensorFlow models called TensorFlow Extended.

The Kubeflow project is dedicated to making deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable.

It is an end-to-end Machine Learning platform for Kubernetes.
It provides components for each stage in the ML lifecycle, starting from exploration of data, model training, and deployment.
Operators can select the best-trained model for the end-users, with no need to deploy every component.

The set of tools available in Kubeflow helps the ML engineers/Data scientists in:
- Data Exploration.
- Build/Train machine learning models.
- Analyze the model performance.
- Hyper-parameter tuning.
- Version different model.
- Manage compute power.
- Serving infrastructure.
- Deploying the best model to production.

It runs on Kubernetes clusters, we can run it either locally or in the cloud.
It boosts the power of training the machine learning models on multiple nodes (i.e., computers).
Reduce the model training time.

The Kubeflow user interface consists of the following:

![image](https://user-images.githubusercontent.com/88195980/192416829-0d666d6e-afc2-43f4-9e91-26d5e846f437.png)


- **Home:** Central Hub to view, access resources recently used, active experiments, and useful documentation.
- **Notebook Servers:** Manage Notebooks servers.
- **TensorBoards:** Manage servers of TensorBoards.
- **Models:** Manage deployed KFServing models.
- **Volumes:** Manage cluster’s Volume.
- **AutoML Experiments:** Manage Katlib experiments.
- **KFP Experiments:** Manage Kubeflow Pipelines (KFP) experiments.
- **Pipelines:** Manage Kubeflow Pipelines.
- **Runs:** Manage KFP runs.
- **Recurring Runs:** Manage KFP recurring runs.
- **Artifacts:** To track ML Metadata (MLMD) artifacts.
- **Execution:** To track various component execution in MLMD.
- **Manage Contributors:** Configure user access sharing across namespaces in the Kubeflow

To access the central dashboard, you need to connect to the Istio gateway that provides access to the Kubeflow service mesh.
How you access the Istio gateway varies depending on how you’ve configured it.

**Kubeflow Notebooks**

Kubeflow Notebooks, a web-based development environment inside our Kubernetes clusters by running them inside the pods.

Users can spin up the notebook servers either using Jupyter lab, R Studio, or Visual Studio Code (code-server).
It can be done directly from the dashboard, allocating the right storage, CPUs, and GPUs.

You can create notebook containers directly in the cluster, rather than locally on their workstations.
Admins can provide standard notebook images for their organization with required packages pre-installed.
Kubeflow’s RBAC can be used to manage the access control that enables easier notebook sharing across the organization.

**ML Libraries and Framework.**

It is compatible with all the required machine learning libraries and frameworks like TensorFlow, PyTorch, XGBoost, sci-kit-learn, MXNet, Keras, and many more.

**Kubeflow Pipelines**

Kubeflow pipeline is a platform for building and deploying scalable, and portable machine learning workflows based on Docker containers.

We can automate our ML workflow into pipelines by containerizing steps as pipeline components and defining inputs, outputs, parameters, and generated artifacts.

So, a big question comes into mind what is a pipeline?
Let’s answer that question first.

**What is Pipeline?**

ML pipeline is a means of automating the machine learning workflow by enabling data to be transformed and correlated into a model that can also be anatomized to achieve outputs. This type of ML pipeline makes the process of inputting data into the ML model completely automated.

ML pipeline is the end-to-end construct that orchestrates the inflow of data into, and output from, a machine learning model (or set of multiple models). It includes raw data input, features, outputs, the machine learning model and model parameters, and prediction outputs.

In Kubeflow, the pipeline component is a self-contained set of user code, packaged as a Docker image, that performs one step in the pipeline. For example, a component can be responsible for data preprocessing, data transformation, model training, and so on.

While writing the code of the pipeline component make sure that all the necessary libraries that are needs to be imported should be defined within the function.

Each pipeline component should be independent of dependencies. This will helps us in many ways.
For example: If we got any failure in the pipeline so, we could easily identify the component which holds the issue and troubleshoot it without impacting other components.

**Katlib for Hyperparameter tuning/AutoML**
- Katlib is the component of Kubeflow that is used for hyperparameter tuning, neural architecture search.
- Katib is a Kubernetes-native project for automated machine learning (AutoML).
- It runs pipelines with different hyperparameters, optimizing for the best ML model.
- Katib is agnostic to machine learning (ML) frameworks.
- It can tune hyperparameters of applications written in any language of the users’ choice and natively supports many ML frameworks, such as TensorFlow, MXNet, PyTorch, XGBoost, and others.

Automated Machine Learning (AutoML) is a way to automate the process of applying machine learning algorithms to solve real-world problems.
Basically, it automates the process of feature selection, composition, and parameterization of machine learning models.

Katlib supports a lot of various AutoML algorithms, such as Bayesian optimization, Tree of Parzen Estimators, Random Search, Covariance Matrix Adaptation Evolution Strategy, Hyperband, Efficient Neural Architecture Search, Differentiable Architecture Search, and many more.

**and many more…**

Kubeflow provides the integration that you need.
It integrates with MLFlow for the model registry, staging, and monitoring in production, Seldon Core for inference serving, and Apache Spark for parallel data processing.

Training of ML models in Kubeflow through operators like TFJobs, PyTorchJob, MXJob, XGBoostJob, and MPIJob.

You can schedule your jobs also with gang scheduling.

![image](https://user-images.githubusercontent.com/88195980/192417289-baf94702-bc94-4c83-bd82-8c827f9ae575.png)




**Let's compare three main tools**


| Name of Service | Additional Info |<IMG  src="https://nub8.net/wp-content/uploads/2019/07/amazon_sagemaker-min.png"  alt="Machine Learning with Amazon SageMaker » Nub8"  width="150" height="50">|<IMG  src="https://miro.medium.com/max/1127/1*-ganvHfXEbn6oYk-krRpIg.jpeg"  alt="Azure Machine Learning Service: Part 1 — An Introduction ..." width="200" height="60"> |<IMG  src="https://vslive.com/-/media/ECG/VSLive/Blogs/AzureDatabricks.jpg"  alt="Azure Databricks: What Is It and What Can You Do with It? -- Visual Studio  Live!: Training Conferences and Events for Enterprise Microsoft .NET and  Azure Developers" width="200" height="75">|
|--|--|:----:|:----:| :---: |
|Notebook Support||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|Jupyter Lab|Azure has entire custom interace|:heavy_check_mark:|:x:|:heavy_check_mark:|
|Computer instance|Azure gives more control over compute|at set-up time|can change via notebook|can be changed
|Workflow pipeline support||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|Studio/Low code/GUI and Drag and Drop support||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|In-build Feature Store||:heavy_check_mark:|:x::heavy_exclamation_mark:Databricks FS can integrated:heavy_exclamation_mark:|:heavy_check_mark:
|Automatic ML||:heavy_check_mark: Auto pilot|:heavy_check_mark: Automated ML|:heavy_check_mark: AutoML|
|Label the data|Both supporting outsorcing|:heavy_check_mark: Ground Truth|:heavy_check_mark:Data Labeling|:heavy_check_mark: Labelbox
|GPU Support||:heavy_check_mark: Framework optimised|:heavy_check_mark:Can install GPU drivers on compute|:heavy_check_mark:
|Real time endpoint||:heavy_check_mark: Internal only via SDK|:heavy_check_mark: Internal via SDK & Public|:heavy_check_mark:
|Offline job||:heavy_check_mark: Batch Transform|:heavy_check_mark: Pipeline|:heavy_check_mark: Batch and streaming
|In-build support for IoT||:heavy_check_mark: Neo|:x::heavy_exclamation_mark:Possible via IoT edge:heavy_exclamation_mark:|:x:
|In-build data visualisation||:x: Almost No or very basic|:heavy_check_mark:|:heavy_check_mark:
|A/B testing support||:heavy_check_mark:Traffic Routing|:x:|:heavy_check_mark:
|Reinforcement Learning||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|Kubernetes Suport||:heavy_check_mark:|:x:|:heavy_check_mark:
|Multiple Model on same endpoint to save cost||:heavy_check_mark:MMS|:x:|:heavy_check_mark:
|Automatic model debugging, tuning||:heavy_check_mark:Debugger Model Tuning|:x:|:heavy_check_mark:
|Auto Scaling, edge optimization of endpoint||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|Model monitoring||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|Responsible AI||:heavy_check_mark:Model Explainability|:heavy_check_mark: Model interpretability|:heavy_check_mark:
|Augmented AI||:heavy_check_mark:|:x:|:x:
  
<!-- blank line -->

-------------------------------------------
<!-- blank line -->
## Sources
- [https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf)
- [https://docs.databricks.com/applications/machine-learning/model-inference/index.html](https://docs.databricks.com/applications/machine-learning/model-inference/index.html)
- [https://medium.com/@jcbaey/azure-databricks-hands-on-6ed8bed125c7](https://medium.com/@jcbaey/azure-databricks-hands-on-6ed8bed125c7)
- [https://learn.microsoft.com/en-us/azure/databricks/scenarios/what-is-azure-databricks-ml](https://learn.microsoft.com/en-us/azure/databricks/scenarios/what-is-azure-databricks-ml)
- [https://learn.microsoft.com/en-us/azure/databricks/scenarios/what-is-azure-databricks](https://learn.microsoft.com/en-us/azure/databricks/scenarios/what-is-azure-databricks)
- [https://medium.com/@vineetjaiswal/introduction-comparison-of-mlops-platforms-aws-sagemaker-azure-machine-learning-gcp-vertex-ai-9c1153399c8e](https://medium.com/@vineetjaiswal/introduction-comparison-of-mlops-platforms-aws-sagemaker-azure-machine-learning-gcp-vertex-ai-9c1153399c8e)
- [https://azure.microsoft.com/en-in/services/machine-learning/#documentation](https://azure.microsoft.com/en-in/services/machine-learning/#documentation)
- [https://towardsdatascience.com/a-brief-introduction-to-azure-machine-learning-studio-9bbf41800a60](https://towardsdatascience.com/a-brief-introduction-to-azure-machine-learning-studio-9bbf41800a60)
- [https://azure.microsoft.com/en-us/products/databricks/](https://azure.microsoft.com/en-us/products/databricks/)
- [https://towardsdatascience.com/aws-sagemaker-db5451e02a79](https://towardsdatascience.com/aws-sagemaker-db5451e02a79)
- [https://docs.databricks.com/](https://docs.databricks.com/)
- [https://medium.com/@knoldus/kubeflow-a-complete-solution-to-mlops-7208deeb80e5](https://medium.com/@knoldus/kubeflow-a-complete-solution-to-mlops-7208deeb80e5)
