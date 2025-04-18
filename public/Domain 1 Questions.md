Q3 
A healthcare company needs to classify human genes into 25 distinct categories based on their genetic characteristics. The company requires a machine learning algorithm that provides clear visibility into how classification decisions are made, allowing them to document the model's internal decision-making process. Which ML algorithm best satisfies these requirements?
Options :
A. Decision trees
B. Linear regression
C. K-means clustering
D. Neural networks		

Correct Answer :
	A. Decision trees
	Explanation :
		Decision trees are the most suitable choice for this scenario because:
		1. Explainability: Decision trees provide clear visibility into how decisions are made through their tree-like structure, making it easy to understand and document the inner mechanism of the model. Each node in the tree represents a decision point based on specific features, making it highly transparent.
		2. Multi-class Classification: The requirement to classify genes into 25 categories is a multi-class classification problem. Decision trees can naturally handle multi-class classification tasks.
		3. Interpretability: Unlike neural networks (which are "black box" models) or linear/logistic regression (which are better for binary classification or continuous outputs), decision trees provide clear decision paths that can be followed from root to leaf, making it easy to explain how the model arrived at its classification.
		The other options are less suitable because:
		- Linear regression (B) is for predicting continuous numerical values
		- K-means clustering (C) is an unsupervised learning algorithm that groups similar data points together based on feature similarity, not appropriate for a supervised classification task where specific gene categories are already defined
		- Examples
			- Heart disease prediction
			- Fraud detection
			- Customer churn (churn/stay)
			- Neural networks (D) are complex "black box" models that lack easy interpretability	
		Resource URL:
			https://d1.awsstatic.com/events/reinvent/2020/Choose_the_right_machine_learning_algorithm_in_Amazon_SageMaker_AIM308.pdf

Q4
A bio-technology startup has developed an image classification model designed to identify plant diseases from photographs of plant leaves. The company needs to evaluate the percentage of leaf images that the model has classified correctly and accurately. Which evaluation metric is most appropriate for measuring this model's performance?
Options :
A. R-squared score
B. Accuracy 
C. Root mean squared error (RMSE)
D. Mean Average Precision (MAP)

Correct Answer :
	B. Accuracy 
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development Lifecycle" ---
	Explanation :
		Accuracy is the most appropriate evaluation metric because:
			1. For classification tasks (like image classification), accuracy measures the proportion of correct predictions
			2. Formula: (True Positives + True Negatives) / Total Predictions
			3. Directly answers "how many images were classified correctly"
			4. Simple to interpret: higher percentage means better performance
		Other options are incorrect because:
			- A: R-squared is for regression problems, measuring variance explanation
			- C: RMSE is for regression problems, measuring prediction error magnitude
			- D: Mean Average Precision (MAP) is primarily used for ranking problems and information retrieval systems, not for basic classification accuracy measurement in this plant disease identification context
		Resource URL:
			https://docs.aws.amazon.com/machine-learning/latest/dg/evaluating-model-accuracy.html
			https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html


Q6
A healthcare company uses Amazon SageMaker in its machine learning pipeline for clinical applications. Their system needs to process medical images with sizes between 500 MB and 1 GB, and inference processing may take up to 1 hour. The company requires predictions to be delivered with near real-time latency. Which SageMaker inference option is most appropriate for these requirements?
Options :
A. Real-time inference
B. Serverless inference
C. Asynchronous inference
D. SageMaker Edge Deployment

Correct Answer :
	C. Asynchronous inference
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3 Describe the ML development Lifecycle ---
	Explanation :
		Asynchronous inference is the correct choice because:
		1. Designed for large payload sizes (up to 1GB)
		2. Handles long processing times (up to 1 hour)
		3. Provides near real-time processing without blocking
		4. Automatically scales and manages queues
		5. Cost-effective for variable workloads
		Other options are incorrect because:
		- A: Real-time inference has payload limitations and is for immediate responses
		- B: Serverless inference has limits on payload size and processing time
		- D: SageMaker Edge Deployment is for running models on edge devices like cameras or sensors, not for handling large processing jobs with near real-time requirements in a cloud environment
		SageMaker Asynchronous Inference is specifically designed for:
		- Large payload sizes (up to 1GB)
		- Long processing times
		- Cost optimization (scales to zero when no requests)
		- Queue-based processing
		- Near real-time requirements with large data
		Resource URL:
			https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html
		

Q7
A company is developing domain-specific AI solutions. Rather than building new models from scratch, they want to leverage knowledge from existing pre-trained models and adapt them for new related tasks. Which machine learning approach best aligns with this strategy?
Options :
A. Increase the number of epochs.
B. Use transfer learning.
C. Decrease the number of epochs.
D. Use distributed training.
	
Correct Answer : 
	B. Use transfer learning
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle." ---
	Explanation :
		Transfer learning is the correct strategy because:
		1. Allows reuse of pre-trained models for new, related tasks
		2. Reduces need to create models from scratch
		3. Leverages existing domain knowledge
		4. More efficient than training new models
		5. Requires less training data and compute resources
		Other options are incorrect because:
		- A: Increasing epochs just extends training time, doesn't adapt existing knowledge
		- C: Decreasing epochs reduces training time but doesn't transfer knowledge
		- D: Distributed training is an approach to accelerate model training by utilizing multiple computing resources in parallel, but doesn't address the requirement of leveraging pre-trained models
		Resource URL:
			https://aws.amazon.com/what-is/transfer-learning/


Q11
A company is developing an ML model using Amazon SageMaker. The AI team needs a solution that allows them to store, share, and manage feature variables across multiple development teams working on the project.
Which SageMaker feature best addresses these requirements?
Options :
A. Amazon SageMaker Feature Store
B. Amazon SageMaker Data Wrangler
C. Amazon SageMaker Clarify
D. Amazon SageMaker Experiments

Correct Answer : 
	A. Amazon SageMaker Feature Store
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation :
		Amazon SageMaker Feature Store is the correct solution because:
		1. Centralized repository for storing and managing ML features
		2. Enables feature sharing across multiple teams
		3. Provides versioning and consistency for features
		4. Allows real-time and batch access to features
		5. Maintains feature definitions and metadata
		Other options are incorrect because:
		- B: Data Wrangler is for data preparation and transformation, not feature sharing
		- C: Clarify is for bias detection and model explainability
		- D: SageMaker Experiments is for tracking, comparing, and evaluating machine learning experiments and model versions, not for storing and sharing feature variables
		Resource URL:
		https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html


Q14
A startup company is creating an educational application where users solve probability-based questions like: "A box contains 8 red, 4 green, and 3 yellow pencils. What is the probability of choosing a green pencil from the box?"  Which approach fulfills these requirements with MINIMAL operational overhead?
Options :
A. Use supervised learning with regression model that will predict probability.
B. Use reinforcement learning to train a model to return the probability.
C. Use code that will calculate probability by using simple rules and computations.
D. Use natural language processing to interpret the question and generate answers.

Correct Answer : 
	C. Use code that will calculate probability by using simple rules and computations.
	Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI"
	Explanation :
		This question tests the ability to identify when AI/ML is NOT the best solution for a problem. For simple mathematical calculations like probability, traditional programming is the most efficient solution because:	
			1. Simple Mathematical Problem
			- Basic probability calculation
			- Can be solved with simple division (4/13 in this case)
			- No need for complex pattern recognition or learning
			1. Operational Overhead Considerations
			- Traditional programming requires minimal infrastructure
			- No model training needed
			- No data collection required
			- Easy to maintain and debug
		Why other options are incorrect:
			A. Supervised Learning (Regression)
			- Unnecessarily complex for simple math
			- Requires training data
			- Higher operational overhead
			- Overkill for basic probability
			B. Reinforcement Learning
			- Designed for decision-making through trial and error
			- Not suitable for fixed mathematical calculations
			- High complexity and resource usage
			D. Natural Language Processing - While NLP could help interpret different question formats, it would require substantial infrastructure, training data, and maintenance compared to simple computational formulas that can handle standardized questions
		Resource URLs:
			https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html

Q15
Which metric evaluates the operational performance efficiency of deployed AI models?
Options :
A. Customer satisfaction score (CSAT)
B. Training time for each epoch
C. Average response time
D. Model convergence rate

Correct Answer :
	C. Average response time
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: Describe the ML development lifecycle." ---
	Explanation :
		Average response time is the correct metric for measuring runtime efficiency because:
		1. Directly measures operational performance of deployed models
		2. Indicates how quickly the model responds to requests
		3. Helps evaluate real-world performance efficiency
		4. Critical for user experience and system performance
		5. Key indicator of model's operational effectiveness
		Other options are incorrect because:
		- A: CSAT measures user satisfaction, not runtime efficiency
		- B: Training time per epoch is a training metric, not operational efficiency
		- D: Model convergence rate relates to how quickly a model reaches optimal parameters during training, not to its performance in production environments
		Resource URL:
			https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html


Q16
A company is developing a customer service center application. The system needs to extract valuable insights from customer conversations. The company requires a solution to analyze audio recordings from customer calls and extract key information. Which AWS service best addresses these requirements?
Options :
A. Build a conversational chatbot by using Amazon Lex.
B. Transcribe call recordings by using Amazon Transcribe.
C. Extract information from call recordings by using Amazon Textract.
D. Create classification labels by using natural language processing Amazon Comprehend.	

Correct Answer :
	B. Transcribe call recordings by using Amazon Transcribe.
	Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI."
	Explanation :
		Amazon Transcribe is the most appropriate solution because:
		1. It's specifically designed to convert speech-to-text from audio recordings
		2. It can process customer call recordings and create accurate transcripts
		3. It supports call analytics features specifically designed for contact centers
		4. The transcribed text can then be used for further analysis and insights extraction
		The solution typically involves using Amazon Transcribe first to convert audio to text, and then potentially using Amazon Comprehend for further text analysis of the transcripts. However, the question specifically asks about analyzing audio content, for which Transcribe is the primary tool needed.
		Other options are incorrect because:
		- A: Amazon Lex is for building conversational interfaces (chatbots), not for analyzing existing conversations
		- C: Amazon Textract is designed for extracting text from documents and images, not for processing audio recordings
		- D: Amazon Comprehend is for natural language processing of text, but cannot process audio directly
		Resource URL:
			https://aws.amazon.com/transcribe/
			https://docs.aws.amazon.com/transcribe/latest/dg/call-analytics.html
		

Q17
A marketing company possesses hundreds of terabytes of unlabeled customer interaction data they want to leverage for targeted product promotions. The company needs to segment its customers into distinct tiers to optimize advertising campaigns for different products. Which machine learning approach should the company implement to address these requirements?
Options :
A. Supervised learning
B. Unsupervised learning
C. Reinforcement learning
D. Semi-supervised learning

Correct Answer :
	B. Unsupervised learning
	Explanation :
		Unsupervised learning is the correct approach for this scenario because:
		1. The company has unlabeled customer data - a key indicator for unsupervised learning use cases
		2. The goal is to classify customers into tiers (clustering) without pre-existing categories
		3. The task involves discovering natural patterns and groupings in customer data for segmentation
		4. The petabyte-scale data can be processed without the need for manual labeling
		Other options are incorrect because:
		- A: Supervised learning requires labeled data for training, which is not available here
		- C: Reinforcement learning is for training agents through reward-based feedback systems
		- D: Semi-supervised learning requires some labeled data to guide the learning process, which isn't mentioned as being available in this scenario
		Resource URL:
		https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/



Q20
A company wants to use AI to safeguard its web application from network-based security threats. The AI solution must be able to determine if incoming traffic from an IP address might be malicious. Which solution would fulfill these requirements?
Options :
A. Create a speech recognition system to detect bad IP.
B. Build a natural language processing (NLP) named entity recognition system.
C. Develop an anomaly detection system.
D. Create a recommendation engine system.
Correct Answer :
	C. Develop an anomaly detection system.
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI" ---
	Explanation:
		An anomaly detection system is the most appropriate solution for identifying suspicious IP addresses because:
		1. Anomaly detection is specifically designed to identify patterns that deviate from normal behavior
		2. It can analyze IP address patterns and flag unusual or suspicious activity in real-time
		3. It's commonly used in security applications to detect potential threats
		4. It can learn from historical IP address data to identify what constitutes normal vs. suspicious behavior
		Other options are incorrect because:
		- A: Speech recognition is for converting spoken words to text, not relevant for IP address analysis
		- B: NLP named entity recognition is for identifying entities in text, not for security threat detection
		- D: A recommendation engine is designed to suggest relevant items to users based on past behavior, not for detecting malicious network activity
		Resource URL:
		https://aws.amazon.com/blogs/machine-learning/detect-suspicious-ip-addresses-with-the-amazon-sagemaker-ip-insights-algorithm/
		


Q27
A company is developing a machine learning model for image classification. Once completed, they plan to deploy the model to production for integration with a web application.
The company needs a solution that can host the ML model and serve predictions without requiring them to manage any underlying infrastructure.
Which solution best meets these requirements?
Options :
A. Use Amazon SageMaker Serverless Inference to deploy the model.
B. Use Amazon CloudFront to deploy the model.
C. Use Amazon API Gateway to host the model and serve predictions.
D. Use Amazon ElastiCache to host the model and serve predictions.

Correct Answer: 
	A. Use Amazon SageMaker Serverless Inference to deploy the model.
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation:
	Amazon SageMaker Serverless Inference is the correct solution because:
	1. Provides serverless deployment (no infrastructure management)
	2. Automatically scales based on traffic
	3. Pay-per-use pricing model
	4. Specifically designed for ML model deployment
	5. Handles model hosting and inference without infrastructure management
	Other options are incorrect because:
	- B: CloudFront is a content delivery network, not for model deployment
	- C: API Gateway manages APIs but doesn't host ML models
	- D: Amazon ElastiCache is a caching service for improving application performance, not designed for hosting or serving machine learning models
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html


Q31
An AI engineer maintains a database containing various wildlife photographs. The engineer aims to create an automated system that can identify different animal species in these images without requiring manual classification.  
Which technique is most appropriate for this task?
Options:
A. Object detection
B. Anomaly detection
C. Named entity recognition
D. Sentiment analysis

Correct Answer:
	A. Object detection
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies" ---
	Explanation:
		Object detection is the correct strategy because:
		1. Specifically designed to identify and classify objects in images
		2. Can automatically detect and categorize animals in photos
		3. Works without human intervention
		4. Provides both location and classification of objects
		5. Ideal for automated image analysis tasks
		Other options are incorrect because:
		- B: Anomaly detection finds unusual patterns, not suitable for classification
		- C: Named entity recognition is for text analysis, not image processing
		- D: Sentiment analysis evaluates emotional tone in text data and has no application for image-based animal identification
		https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html
		https://docs.aws.amazon.com/sagemaker/latest/dg/algo-object-detection-tech-notes.html


Q36
A company developed a deep learning model for object shape recognition. After the AI team deployed the model to the production environment, users began submitting new images for analysis. Which AI process is taking place when the model examines these new images to recognize object shapes?
Options :
A. Training
B. Inference
C. Model deployment
D. Bias correction

Correct Answer: 
	B. Inference
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies" ---
	Explanation:
	Inference is the correct answer because:
	1. It's the process of using a trained model to make predictions
	2. Occurs when model analyzes new, unseen data
	3. Represents the production use of the model
	4. Takes place after model training and deployment
	5. Involves applying learned patterns to new inputs
	Other options are incorrect because:
	- A: Training is the process of teaching the model, not using it
	- C: Model deployment is setting up the model for use, not using it
	- D: Feature extraction is a component of the model's processing pipeline but not the overall process of generating predictions from new inputs
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html

Q44
A company has built a prediction model for retail product pricing. The model demonstrated high accuracy in the test environment, but after deployment to production, its performance deteriorated considerably.  
What strategy should the company implement to resolve this problem?
Options:
A. Reduce the volume of data that is used in training.
B. Implement cross-validation techniques.
C. Increase the volume of data that is used in training.
D. Increase the model training time.

Correct Answer: 
	C. Increase the volume of data that is used in training.
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies" ---
	Explanation:
		Increasing the volume of training data is the correct solution because:
		1. Helps prevent overfitting (when model performs well on training but poorly on new data)
		2. Improves model generalization
		3. Provides more diverse examples for learning
		4. Enhances model robustness
		5. Reduces the gap between training and production performance
		Other options are incorrect because:
		- A: Reducing training data would likely worsen the problem
		- B: While cross-validation is useful for model evaluation, it doesn't directly address the fundamental issue of limited training data diversity needed for production scenarios
		- D: Increasing training time alone doesn't improve generalization
	Resource URL:
	https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

Q103
A data scientist is working on a solution to categorize flower species using measurements of petal length, petal width, sepal length, and sepal width.  
Which algorithm is most appropriate for this requirement?
Options:
A. K-nearest neighbors (k-NN)
B. K-mean
C. Principal Component Analysis (PCA)
D. Linear regression

Correct Answer:
	A. K-nearest neighbors (k-NN)
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation:
		Answer A is correct because:
		- K-nearest neighbors (k-NN) is a supervised learning algorithm ideal for classification problems
		- It works well with multiple input features (in this case, four flower measurements)
		- K-NN is particularly effective for pattern recognition in datasets where classes can be distinguished by feature measurements
		- The algorithm is well-suited for the famous Iris flower dataset, which is a common example of this exact use case
		- K-NN makes predictions based on the similarity of features to known examples, which is perfect for flower classification
		The other answers are not suitable because:
		B. K-means
		- This is an unsupervised clustering algorithm
		- While it could group similar flowers, it doesn't use labeled data for classification
		- Not ideal for predictive classification tasks where classes are known
		C. Autoregressive Integrated Moving Average (ARIMA)
		- PCA is a dimensionality reduction technique rather than a classification algorithm, so it cannot directly predict flower species from measurements
		D. Linear regression
		- Used for predicting continuous numerical values
		- Not appropriate for classification tasks
		- Outputs continuous values rather than discrete classes
	Resource URLs:
		https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html
		https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html
		https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-arima.html
		https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html

Q109
A company is developing a machine learning model to identify potential customer turnover. The model shows excellent accuracy with the training data but fails to correctly predict churn patterns in new data.

Which approach would address this problem?
Options:
A. Decrease the regularization parameter to increase model complexity.
B. Increase the regularization parameter to decrease model complexity.
C. Add more features to the input data.
D. Remove all preprocessing steps from the data pipeline.

Let me help classify and explain this question:

Correct Answer: 
	B. Increase the regularization parameter to decrease model complexity.
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation: 
		Answer B is correct because:
		- The scenario describes a classic case of overfitting, where the model performs well on training data but poorly on new data
		- Increasing the regularization parameter helps reduce overfitting by decreasing model complexity
		- Regularization penalizes large parameter values, forcing the model to find simpler, more generalizable solutions
		- A simpler model is more likely to perform well on new, unseen data
		The other answers are not suitable because:
		A. Decreasing regularization would make overfitting worse by allowing the model to become more complex
		C. Adding more features could potentially increase overfitting if the features aren't relevant
		D. Removing preprocessing steps would likely reduce model performance as proper data preprocessing is essential for effective machine learning models
	Resource URL:
	https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

Q114
A company wants to build an ML application.  
Order the correct steps from the following list to develop a well-architected ML workload. 
Hotspot : 1. Deploy model - 2. Develop model - 3. Monitor model - 4.Define business goal and frame ML problem
Options:
A. 1,2,3,4
B. 4,2,1,3
C. 4,1,2,3
D. 2,4,1,3

Correct Answer: 
	B. 4,2,1,3 (Define business goal, Develop model, Deploy model, Monitor model)
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
		Answer B is correct because:
		- The ML development lifecycle follows a logical sequence that starts with business understanding and ends with monitoring
		- Step 4 (Define business goal) must come first to ensure the ML solution addresses the right problem
		- Step 2 (Develop model) follows as you build the solution to address the defined problem
		- Step 1 (Deploy model) comes after development when the model is ready for production
		- Step 3 (Monitor model) is the final ongoing phase to ensure continued performance
		The other answers are not suitable because:
		A. 1,2,3,4 - Starts with deployment before defining the problem or developing the model
		C. 4,1,2,3 - Deploys the model before development
		D. 2,4,1,3 - Develops the model before defining the business goal
	Resource URL:
	https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/best-practices-by-ml-lifecycle-phase.html

Q116
A business must develop an ML model to categorize images containing various animal species. The business possesses an extensive dataset of pre-labeled images and doesn't intend to perform additional labeling.  
Which learning approach should the business implement for model training?
Options:
A. Supervised learning
B. Unsupervised learning
C. Reinforcement learning
D. Transfer learning

Correct Answer: 
	A. Supervised learning
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation: 
		Answer A is correct because:
		- The business has labeled images (labeled data)
		- The task is classification, which is a typical supervised learning problem
		- Supervised learning is used when you have input data (images) and corresponding output labels (animal types)
		- The dataset is already labeled and no more labeling is planned, making supervised learning the most appropriate approach
		The other answers are not suitable because:
		B. Unsupervised learning - Used when data is unlabeled and the goal is to find patterns/clusters
		C. Reinforcement learning - Used for learning through interaction with an environment and receiving rewards/penalties
		D. Transfer learning - While useful for image tasks, it's a technique for leveraging pre-trained models rather than a learning paradigm, and doesn't address the fundamental approach needed for the labeled dataset
	Resource URL:
	https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/

Q117
At which phase of the ML lifecycle are compliance and regulatory constraints typically established?
Options:
A. Feature engineering
B. Model evaluation
C. Data collection
D. Business goal identification

Correct Answer: 
	D. Business goal identification
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
		Answer D is correct because:
		- Business goal identification is the initial phase where all requirements, including compliance and regulatory needs, should be defined
		- Compliance and regulatory requirements must be identified early to ensure they're incorporated throughout the entire ML lifecycle
		- Understanding regulatory requirements during business goal identification helps shape:
		  * Data collection strategies
		  * Model development approaches
		  * Implementation methods
		  * Monitoring requirements
		The other answers are not suitable because:
		A. Feature engineering - This is a technical phase focused on creating model inputs, too late for determining compliance requirements
		B. Model evaluation - This phase focuses on assessing model performance against metrics, not establishing regulatory requirements
		C. Data collection - While compliance affects data collection, the requirements should be identified before starting data collection
	Resource URL:
	https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/best-practices-by-ml-lifecycle-phase.html

Q118
A F&B sector enterprise needs to develop an ML model to reduce daily food wastage and boost sales performance. The enterprise requires ongoing model accuracy improvements.  
Which solution satisfies these requirements?
Options:
A. Use Amazon SageMaker and iterate with newer data.
B. Use Amazon Personalize and iterate with historical data.
C. Use Amazon Athena to analyze customer orders.
D. Use Amazon Rekognition to optimize the model.

Correct Answer: 
	A. Use Amazon SageMaker and iterate with newer data
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
	Answer A is correct because:
	- Amazon SageMaker provides the complete ML development lifecycle management
	- It allows continuous model improvement through retraining with new data
	- SageMaker supports iterative model development and deployment
	- The solution addresses both the need for prediction (food waste/sales) and continuous improvement
	The other answers are not suitable because:
	B. Amazon Personalize - This is specifically for recommendation systems, not for general prediction problems like food waste
	C. Amazon Athena - This is a serverless query service for analyzing data in S3, not an ML development platform for building predictive models
	D. Amazon Rekognition - This is specifically for image and video analysis, not for business prediction problems
	Resource URL:
	https://aws.amazon.com/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/

Q119
A company has built an ML model for predicting property resale values. The company needs to deploy the model for inference without managing any infrastructure.

Which solution best meets these requirements?
Options:
A. Deploy the model on an Amazon EC2 instance.
B. Deploy the model on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster.
C. Deploy the model by using Amazon CloudFront with an Amazon S3 integration.
D. Deploy the model by using an Amazon SageMaker endpoint.

Correct Answer: 
	D. Deploy the model by using an Amazon SageMaker endpoint
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
		Answer D is correct because:
		- Amazon SageMaker endpoints provide serverless model deployment
		- It automatically handles infrastructure management
		- Provides scalable, fully-managed inference endpoints
		- Requires no server management from the user
		- Aligns with the requirement of making predictions without managing infrastructure
		The other answers are not suitable because:
		A. Amazon EC2 instance - Requires manual server management and infrastructure maintenance
		B. Amazon EKS cluster - Requires cluster management and infrastructure configuration
		C. CloudFront with S3 - These services are for content delivery and storage, not ML model deployment
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html

Q121
A manufacturing enterprise implements AI to assess product quality and identify any flaws or defects. Which category of AI technology is the enterprise utilizing?
Options:
A. Recommendation system
B. Natural language processing (NLP)
C. Computer vision
D. Anomaly detection

Correct Answer: 
	C. Computer vision
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation: 
		Answer C is correct because:
		- Computer vision is specifically designed for machines to understand and analyze visual information
		- It's the primary AI technology used for visual inspection tasks
		- Computer vision can detect defects, damages, and anomalies in products
		- This technology enables automated quality control through visual inspection
		- It's commonly used in manufacturing for product quality assurance
		The other answers are not suitable because:
		A. Recommendation system - Used for suggesting items/content, not for visual inspection
		B. Natural language processing (NLP) - Deals with text and speech, not visual inspection
		D. Anomaly detection - While related to identifying unusual patterns, this is a technique used across multiple AI domains and isn't specific to visual inspection of products
	Resource URL:
		https://aws.amazon.com/what-is/computer-vision/


Q122
Some retail company wants to create and develop ML model to predict customer satisfaction. The company needs to speed up the tuning by fully automating the model.  
Which AWS service meets these requirements?
Options:
A. Amazon Personalize
B. Amazon SageMaker
C. Amazon QuickSight
D. Amazon Comprehend

Correct Answer: 
	B. Amazon SageMaker
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
		Answer B is correct because:
		- Amazon SageMaker provides automated model tuning through SageMaker Automatic Model Tuning
		- It can automatically find the best hyperparameters for any ML model
		- SageMaker supports end-to-end ML development, including automated tuning capabilities
		- It can handle customer satisfaction prediction models through various algorithms
		- Offers built-in hyperparameter optimization (HPO) functionality
		The other answers are not suitable because:
		A. Amazon Personalize - Specifically for recommendation systems, not general prediction tasks
		C. Amazon QuickSight - This is a business intelligence and data visualization service that doesn't provide ML model development or automated tuning capabilities
		D. Amazon Comprehend - This is specifically for natural language processing tasks
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html

Q127
An ecommerce company wants to enhance search engine functionality recommendations by customizing the results for each user of the company’s ecommerce platform.  
Which AWS service meets these requirements?
Options:
A. Amazon Personalize
B. Amazon Kendra
C. Amazon Rekognition
D. Amazon Lex


Correct Answer: 
	A. Amazon Personalize
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation: 
	Answer A is correct because:
	- Amazon Personalize is specifically designed for creating personalized recommendations
	- It can customize search results based on individual user behavior
	- Perfect for e-commerce applications requiring personalized user experiences
	- Can provide real-time personalization for search results
	- Automatically learns from user interactions to improve recommendations
	The other answers are not suitable because:
	B. Amazon Kendra - Enterprise search service, but doesn't provide personalized recommendations
	C. Amazon Rekognition - For image and video analysis, not for search recommendations
	D. Amazon Lex - A service for building conversational interfaces using voice and text, not designed for personalizing search results based on user behavior in an e-commerce context
	Resource URL:
	https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html

Q129
A corporation requires performance tracking of its machine learning deployments using a highly scalable AWS offering.  
Which AWS service satisfies these specifications?
Options:
A. Amazon CloudWatch
B. AWS CloudTrail
C. AWS Trusted Advisor
D. AWS Health Dashboard

Correct Answer: 
	A. Amazon CloudWatch
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
	Answer A is correct because:
	- Amazon CloudWatch is specifically designed for monitoring AWS resources and applications
	- It provides scalable monitoring capabilities for ML systems
	- Offers real-time monitoring and metrics collection
	- Can track ML model performance metrics
	- Supports automated actions based on performance thresholds
	- Highly scalable and integrates well with ML services
	The other answers are not suitable because:
	B. AWS CloudTrail - For logging API activity, not for performance monitoring
	C. AWS Trusted Advisor - For optimization recommendations, not for performance monitoring
	D. AWS Health Dashboard - Designed to provide visibility into the availability of AWS services and account-specific notifications, not specifically for monitoring ML system performance metrics
	https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html


Q132
A business is developing a smartphone application for visually impaired individuals. The application needs to recognize spoken commands and communicate back with audible responses.  
Which approach will best fulfill these requirements?
Options:
A. Use a deep learning neural network to perform speech recognition.
B. Build ML models to analyze textual sentiment patterns.
C. Use generative AI summarization to generate human-like text.
D. Build custom models for image classification and recognition.

Correct Answer: 
	A. Use a deep learning neural network to perform speech recognition
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation: 
		Answer A is correct because:
		- Speech recognition using deep learning is ideal for processing user voice input
		- Deep learning neural networks can handle both speech-to-text and text-to-speech conversion
		- This solution addresses both requirements: hearing users and providing voice responses
		- It's particularly suitable for accessibility applications
		- Neural networks are proven effective for speech processing tasks
		The other answers are not suitable because:
		B. ML models for textual sentiment patterns - This focuses on sentiment analysis rather than speech interaction, which doesn't address the core requirements of processing spoken commands and generating voice responses
		C. Generative AI summarization - While it generates text, it doesn't address speech processing needs
		D. Image classification models - Not relevant for speech-based interaction requirements
	Resource URL:
	https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
	https://docs.aws.amazon.com/polly/latest/dg/what-is.html

Q135
A company wants to develop ML applications to improve business operations and efficiency.  
Select the correct ML paradigm (supervised or unsupervised) from the following list for each use case. Hotspot Use case :
1) Binary classification
2) Multi-class classification
3) K-means clustering
4) Dimensionality reduction

Options:
A. 1 Supervised 2. Supervised 3. Unsupervised 4. Unsupervised
B. 1 Unsupervised 2. Supervised 3. Unsupervised 4. Unsupervised
C. 1 Unsupervised 2. Unsupervised 3. Supervised 4. Supervised
D. 1 Supervised 2. Supervised 3. Unsupervised 4. Supervised


Correct Answer: 
	A. 1. Supervised 2. Supervised 3. Unsupervised 4. Unsupervised
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation: 
		Answer A is correct because:
		1. Binary Classification (Supervised):
		- Has labeled data with two classes
		- Requires known outcomes for training
		- Examples: spam/not spam, fraud/not fraud
		2. Multi-class Classification (Supervised):
		- Has labeled data with multiple classes
		- Requires known outcomes for training
		- Examples: image classification, sentiment analysis
		3. K-means Clustering (Unsupervised):
		- No labeled data needed
		- Finds natural groupings in data
		- Groups similar data points together
		4. Dimensionality Reduction (Unsupervised):
		- No labeled data needed
		- Reduces number of features while preserving information
		- Examples: PCA, t-SNE
		The other answers are not suitable because they incorrectly classify one or more ML paradigms.
	Resource URL:
	https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/



Q138
A retail business needs to segment its clientele based on customer characteristics and purchasing behaviors.  
Which algorithm would be most appropriate to address this requirement?
Options :
A. K-nearest neighbors (k-NN)
B. K-means
C. Decision tree
D. Random forest

Correct Answer: 
	B. K-means
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---	
	Explanation: 
		Answer B is correct because:
		- K-means is specifically designed for clustering/grouping similar data points
		- It can identify natural groupings in customer data based on multiple features
		- Perfect for customer segmentation based on demographics and behavior
		- Unsupervised learning approach that doesn't require labeled data
		- Effectively handles numerical features like demographics and purchase patterns
		The other answers are not suitable because:
		A. K-nearest neighbors (k-NN) - This is a supervised learning algorithm for classification, not for finding groups
		C. Decision tree - This is a supervised learning algorithm for classification and regression
		D. Random forest - This is an ensemble supervised learning method used primarily for classification and regression tasks, not for discovering natural groupings in unlabeled data
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html

Q151
A business needs to detect offensive text in the comments area of its social platform by implementing an ML solution. The business does not have access to labeled data for model training.  
Which approach should the business adopt to identify inappropriate content?
Options:
A. Use Amazon Rekognition moderation.
B. Use Amazon Comprehend toxicity detection.
C. Use Amazon SageMaker built-in algorithms to train the model.
D. Use Amazon Textract to analyze comments.

Correct Answer: 
	B. Use Amazon Comprehend toxicity detection
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation: 
		Answer B is correct because:
		- Amazon Comprehend has built-in toxicity detection capabilities
		- Doesn't require labeled training data (meets the requirement)
		- Specifically designed for text analysis and content moderation
		- Can automatically identify harmful language in text
		- Ready-to-use service without need for model training
		The other answers are not suitable because:
		A. Amazon Rekognition - For image and video analysis, not text content
		C. SageMaker built-in algorithms - Would require labeled training data
		D. Amazon Textract - Designed for extracting text and data from scanned documents, not for analyzing sentiment or detecting harmful content in text that's already digital
	Resource URL:
	https://docs.aws.amazon.com/comprehend/latest/dg/toxicity.html

Q152
A streaming service wants to examine audience viewing patterns and characteristics to suggest tailored programming. The company wants to implement a specially developed ML model in its live environment. The company also wants to track if the model effectiveness degrades with time.  
Which AWS service or feature satisfies these requirements?
Options:
A. Amazon Rekognition
B. Amazon SageMaker Clarify
C. Amazon Personalize
D. Amazon SageMaker Model Monitor

Correct Answer: 
	D. Amazon SageMaker Model Monitor
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle" ---
	Explanation: 
	Answer D is correct because:
	- SageMaker Model Monitor specifically tracks model quality and drift over time
	- Designed for monitoring models in production environments
	- Can detect data and model drift
	- Provides continuous monitoring capabilities
	- Helps maintain model quality in production
	The other answers are not suitable because:
	A. Amazon Rekognition - For image and video analysis, not model monitoring
	B. SageMaker Clarify - For bias detection and model explainability, not continuous monitoring
	C. Amazon Personalize - Provides recommendation systems but lacks built-in capabilities for monitoring model quality degradation over time, which is the key requirement for detecting model drift
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html


Q154
A production company needs to generate product specifications in multiple languages.  
Which AWS service will best automate this requirement?
Options:
A. Amazon Translate
B. Amazon Transcribe
C. Amazon Kendra
D. Amazon Comprehend

Correct Answer: 
	A. Amazon Translate
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation: 
	Answer A is correct because:
	- Amazon Translate is specifically designed for translating text between languages
	- It can automatically create product descriptions in multiple languages
	- Supports real-time and batch translation
	- Ideal for multilingual content creation
	- Provides neural machine translation capabilities
	The other answers are not suitable because:
	B. Amazon Transcribe - Converts speech to text, not for language translation
	C. Amazon Kendra - Enterprise search service, not for translation
	D. Amazon Comprehend - Natural language processing service for extracting insights and relationships from text, not designed for translating content between languages
	Resource URL:
	https://docs.aws.amazon.com/translate/latest/dg/what-is.html

Q56
A hardware manufacturing company wants to forecast consumer demand for storage components. The company lacks programming skills and machine learning expertise but needs to create a data-driven prediction model. The company needs to analyze both proprietary data and market data.  
Which solution is most appropriate for these requirements?
Options:
A. Store the data in Amazon S3. Create ML models and demand forecast predictions by using Amazon SageMaker built-in algorithms that use the data from Amazon S3.
B. Import the data into Amazon SageMaker Data Wrangler. Create ML models and demand forecast predictions by using SageMaker built-in algorithms.
C. Import the data into Amazon SageMaker Data Wrangler. Build ML models and demand forecast predictions by using an Amazon Forecast AutoPredictor.
D. Import the data into Amazon SageMaker Canvas. Build ML models and demand forecast predictions by selecting the values in the data from SageMaker Canvas.

Correct Answer:
	D. Import the data into Amazon SageMaker Canvas. Build ML models and demand forecast predictions by selecting the values in the data from SageMaker Canvas.
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation:
	Answer D is correct because:
	1. Amazon SageMaker Canvas is specifically designed for users without coding experience or deep ML knowledge, which aligns perfectly with the company's situation.
	2. SageMaker Canvas allows users to build ML models and generate predictions using a visual, point-and-click interface, making it ideal for the company's needs.
	3. It supports importing data from various sources, including both internal and external data, which meets the company's requirement to perform analysis on both types of data.
	4. SageMaker Canvas can be used to create demand forecasting models, which is exactly what the digital devices company needs for predicting customer demand for memory hardware.
	The other answers are not suitable/not relevant because:
	A. While this solution could work, it requires more technical knowledge and coding experience to use Amazon SageMaker directly, which the company lacks.
	B. This solution also requires more technical expertise to use SageMaker built-in algorithms, which doesn't align with the company's lack of coding experience and ML knowledge.
	C. While Amazon Forecast is designed for time-series forecasting, using it through Data Wrangler still requires coding expertise and ML knowledge that the company doesn't possess, making it less suitable than the no-code SageMaker Canvas option.
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/canvas.html

Q58
A company is developing a customer service virtual assistant. The company wants the assistant to enhance its responses by learning from historical conversations and web-based knowledge sources.  
Which AI learning approach enables this autonomous improvement capability?
Options:
A. Supervised learning with a manually curated dataset of good responses and bad responses
B. Reinforcement learning with rewards for positive customer feedback
C. Unsupervised learning to find clusters of similar customer inquiries
D. Transfer learning with pre-trained language models

Correct Answer:
	B. Reinforcement learning with rewards for positive customer feedback
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation:
	Answer B is correct because:
	Reinforcement learning is an AI learning strategy that allows an agent (in this case, the chatbot) to learn from its interactions with the environment (customer interactions) and improve its performance over time. This approach aligns perfectly with the company's requirement for the chatbot to improve its responses by learning from past interactions.
	1. The chatbot can receive rewards for positive customer feedback, encouraging it to repeat successful interactions.
	2. It can learn from both past interactions and potentially incorporate information from online resources to improve its responses.
	3. This approach allows for continuous learning and adaptation, which is essential for a customer service chatbot that needs to handle evolving customer inquiries.
	The other answers are not suitable/not relevant because:
	A. Supervised learning with a manually curated dataset: This approach doesn't allow for continuous improvement based on new interactions. It relies on pre-labeled data and doesn't provide the self-improvement capability required.
	C. Unsupervised learning to find clusters: While this could be useful for organizing customer inquiries, it doesn't directly improve the chatbot's responses or provide a mechanism for learning from past interactions.
	D. Transfer learning with pre-trained language models: While this leverages existing knowledge from pre-trained models, it doesn't provide the feedback-based improvement mechanism needed for the chatbot to learn from its own interactions and customer feedback.
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/reinforcement-learning.html

Q59
An AI engineer has created a neural network model for identifying different materials within photography. The engineer is seeking to evaluate model effectiveness.  
Which measurement tool would most assist the AI professional in assessing the model's classification capability?
Options:
A. Confusion matrix
B. Covariance matrix
C. R2 score
D. Mean squared error (MSE)

Correct Answer:
	A. Confusion matrix
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle." ---
	Explanation:
	Answer A is correct because:
	A confusion matrix is the most appropriate metric for evaluating classification model performance because:
	1. It provides a detailed breakdown of correct and incorrect predictions for each class
	2. It shows true positives, true negatives, false positives, and false negatives
	3. From the confusion matrix, you can derive important classification metrics like:
	   * Accuracy
	   * Precision
	   * Recall
	   * F1-score
	4. It's specifically designed for classification problems, which matches the scenario of classifying materials in images
	The other answers are not suitable/not relevant because:
	B. Covariance matrix: This statistical tool shows how variables vary together and their relationship strength, but doesn't evaluate classification performance or prediction accuracy.
	C. R2 score: This is a metric for regression problems that measures how well the model fits the data. It's not appropriate for classification tasks.
	D. Mean squared error (MSE): This is also a metric for regression problems that measures the average squared difference between predicted and actual values. It's not suitable for classification tasks.
	Resource URLs:
	https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html
	https://docs.aws.amazon.com/machine-learning/latest/dg/evaluating-model-performance.html

Q62
A startup company needs to develop an ML model to analyze historical data archives. The company must run inference on large datasets measuring multiple GBs in size. The company doesn't need immediate access to the model's predictions.
Which Amazon SageMaker inference option will meet these requirements most effectively?
Options:
A. Batch transform
B. Real-time inference
C. Serverless inference
D. Multimodel endpoints

Correct Answer:
	A. Batch transform
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.3: "Describe the ML development lifecycle." ---
	Explanation:
		Answer A is correct because:
		Amazon SageMaker Batch Transform is specifically designed for:
		1. Processing large datasets in batches
		2. Analyzing archived data without real-time requirements
		3. Running predictions on multiple GB-sized datasets efficiently
		4. Scenarios where immediate results are not needed
		5. Cost-effective processing of large volumes of data
		The other answers are not suitable/not relevant because:
		B. Real-time inference: This is designed for scenarios requiring immediate responses and is not cost-effective for large batch processing of archived data.
		C. Serverless inference: While this automatically manages infrastructure, it's better suited for workloads with intermittent traffic patterns and smaller payload sizes.
		D. Multimodel endpoints: This option allows hosting multiple models on a single endpoint to share computing resources, but it's designed for reducing costs when deploying multiple models, not specifically for processing large batches of data. It doesn't address the requirement for processing large datasets in a non-time-sensitive manner.
	Resource URLs:
	https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html


Q68
A company manually examines all submitted resumes in PDF format. As the business grows rapidly, the company anticipates the number of resumes will surpass their review capacity. The company needs an automated solution to convert PDF resumes into plain text format for further processing.
Which AWS service meets this requirement?
Options:
A. Amazon Textract
B. Amazon Personalize
C. Amazon Comprehend
D. Amazon Transcribe

Correct Answer:
	A. Amazon Textract
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.2: "Identify practical use cases for AI." ---
	Explanation:
	Answer A is correct because:
	Amazon Textract is specifically designed to:
	1. Extract text, forms, and data from scanned documents including PDFs
	2. Convert document text into machine-readable format
	3. Maintain text formatting and structure during extraction
	4. Handle large volumes of documents efficiently
	5. Process multiple types of documents including resumes
	The other answers are not suitable/not relevant because:
	B. Amazon Personalize - This service is for creating personalized recommendations and is not related to document text extraction.
	C. Amazon Comprehend - This service analyzes text to extract insights and relationships in content, but it doesn't convert PDF documents to text format. It works with text that's already been extracted, making it unsuitable for the initial PDF conversion requirement.
	D. Amazon Transcribe - This service converts speech to text (audio/video to text) and is not designed for extracting text from PDF documents.
	Resource URL:
	https://docs.aws.amazon.com/textract/latest/dg/what-is.html
