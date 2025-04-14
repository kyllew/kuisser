Q3 
A healthcare company wants to classify human genes into 25 categories based on gene characteristics. The company requirement is to have ML algorithm to document how the inner mechanism of the model affects the output. Which ML algorithm meets these requirements?
Options :
A. Decision trees
B. Linear regression
C. Logistic regression
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
		- Logistic regression (C) is primarily for binary classification
		- Examples
			- Heart disease prediction
			- Fraud detection
			- Customer churn (churn/stay)
			- Neural networks (D) are complex "black box" models that lack easy interpretability	
		Resource URL:
			https://d1.awsstatic.com/events/reinvent/2020/Choose_the_right_machine_learning_algorithm_in_Amazon_SageMaker_AIM308.pdf

Q4
A bio-technology startup has built an image classification model. The model will be used to predict plant diseases from photos of plant leaves. The company wants to evaluate how many images the model classified correctly.  Which evaluation metric should the company use to measure the model's performance?
Options :
A. R-squared score
B. Accuracy 
C. Root mean squared error (RMSE)
D. Learning rate

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
			- D: Learning rate is a training hyperparameter, not an evaluation metric
		Resource URL:
			https://docs.aws.amazon.com/machine-learning/latest/dg/evaluating-model-accuracy.html
			https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html


Q6
Example company uses Amazon SageMaker for its ML pipeline for their production application. The application will process large input data sizes more than 500 MB and up to 1 GB and processing times up to 1 hour. The company stated that it needs near real-time latency.  Which SageMaker inference option can be proposed to achieve this requirements?
Options :
A. Real-time inference
B. Serverless inference
C. Asynchronous inference
D. Batch transform

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
		- D: Batch transform is for offline processing, not near real-time requirements
		SageMaker Asynchronous Inference is specifically designed for:
		- Large payload sizes (up to 1GB)
		- Long processing times
		- Cost optimization (scales to zero when no requests)
		- Queue-based processing
		- Near real-time requirements with large data
		Resource URL:
			https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html
		

Q7
A company is using domain-specific models. The objectives for the project is to avoid creating new models from the beginning. The company instead wants to adapt pre-trained models to create models for new, related tasks.  Which ML strategy meets these requirements?
Options :
A. Increase the number of epochs.
B. Use transfer learning.
C. Decrease the number of epochs.
D. Use unsupervised learning.
	
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
		- D: Unsupervised learning is a different learning approach, not related to model adaptation
		Resource URL:
			https://aws.amazon.com/what-is/transfer-learning/


Q11
A company wants to build an ML model by using Amazon SageMaker. The company needs to share and manage variables for model development across multiple teams.  
Which SageMaker feature meets these requirements?
Options :
A. Amazon SageMaker Feature Store
B. Amazon SageMaker Data Wrangler
C. Amazon SageMaker Clarify
D. Amazon SageMaker Model Cards

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
		- D: Model Cards are for model documentation and governance
		Resource URL:
		https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html


Q14
Startup company wants to develop an educational game where users answer questions such as the following: "A box contains 8 red, 4 green, and 3 yellow pencils. What is the probability of choosing a green pencil from the box?"  Which solution meets these requirements with the LEAST operational overhead?
Options :
A. Use supervised learning with regression model that will predict probability.
B. Use reinforcement learning to train a model to return the probability.
C. Use code that will calculate probability by using simple rules and computations.
D. Use unsupervised learning to create a model that will estimate probability density.	

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
			D. Unsupervised Learning
			- Used for finding patterns in unlabeled data
			- Not appropriate for simple probability calculations
			- Adds unnecessary complexity
		Resource URLs:
			https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html

Q15
Which metric measures the runtime efficiency of operating AI models?
Options :
A. Customer satisfaction score (CSAT)
B. Training time for each epoch
C. Average response time
D. Number of training instances

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
		- D: Number of training instances relates to training capacity, not runtime performance
		Resource URL:
			https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html


Q16
A company is planning to build a support and contact center application. The application also will be able to gain insights from customer conversations. The company wants to analyze and extract key information from the audio of the customer calls. Which solution meets these requirements?
Options :
A. Build a conversational chatbot by using Amazon Lex.
B. Transcribe call recordings by using Amazon Transcribe.
C. Extract information from call recordings by using Amazon Poly.
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
		- C: SageMaker Model Monitor is for monitoring ML model performance, not for audio analysis
		- D: Amazon Comprehend is for natural language processing of text, but cannot process audio directly
		Resource URL:
			https://aws.amazon.com/transcribe/
			https://docs.aws.amazon.com/transcribe/latest/dg/call-analytics.html
		

Q17
A advertisement company has hundreds of terabytes unlabeled customer data to use for some product campaign. The company wants to classify its customers into tiers to advertise and promote the company's products. Which methodology should the company use to meet these requirements?
Options :
A. Supervised learning
B. Unsupervised learning
C. Reinforcement learning
D. Reinforcement learning from human feedback (RLHF)

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
		- D: RLHF is specifically for fine-tuning models based on human feedback, not for customer segmentation
		Resource URL:
		https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/



Q20
A company wants to use AI to protect its application from web attack network threats. The AI solution needs to check if an IP address is coming from a suspicious source. Which solution that will satisfy requirements?
Options :
A. Create a speech recognition system to detect bad IP.
B. Build a natural language processing (NLP) named entity recognition system.
C. Develop an anomaly detection system.
D. Create a fraud forecasting system.
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
		- D: Fraud forecasting is predictive in nature and typically used for financial transactions, not real-time IP threat detection
		Resource URL:
		https://aws.amazon.com/blogs/machine-learning/detect-suspicious-ip-addresses-with-the-amazon-sagemaker-ip-insights-algorithm/
		


Q27
A company has a project to develop ML model for image classification. The company wants to deploy the model to production so that a web application can use the model.  
The company needs to implement a solution to host the model and serve predictions without managing any of the underlying infrastructure.  
Which solution will meet these requirements?
Options :
A. Use Amazon SageMaker Serverless Inference to deploy the model.
B. Use Amazon CloudFront to deploy the model.
C. Use Amazon API Gateway to host the model and serve predictions.
D. Use AWS Batch to host the model and serve predictions.

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
	- D: AWS Batch is for batch processing, not real-time model serving
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html


Q31
An AI engineer has a collection of animal photos placed in database. The engineer wants to automatically identify and categorize the animals in the photos without manual human effort.  
Which strategy meets these requirements?
Options:
A. Object detection
B. Anomaly detection
C. Named entity recognition
D. Inpainting

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
		- D: Inpainting is for filling in missing or damaged parts of images
		Resource URL:
		https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html
		https://docs.aws.amazon.com/sagemaker/latest/dg/algo-object-detection-tech-notes.html


Q36
A company built a deep learning model for object detection. They wanted to detect the shape of the object at initial phase. The AI team deployed the model to production. Which AI process occurs when the model analyzes a new image to identify objects?
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
	- D: Bias correction is part of model improvement, not prediction
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html

Q44
A company is developing a new model to predict the prices of specific items. The model performed well on the training dataset. When the company deployed the model to production, the model's performance decreased significantly.  
What should the company do to mitigate this problem?
Options:
A. Reduce the volume of data that is used in training.
B. Add hyperparameters to the model.
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
		- B: Adding hyperparameters doesn't address generalization
		- D: Increasing training time alone doesn't improve generalization
	Resource URL:
	https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

Q103
An AI engineer involved in project to predict the classification of flowers based on petal length, petal width, sepal length, and sepal width.  
Which algorithm meets these requirements?
Options:
A. K-nearest neighbors (k-NN)
B. K-mean
C. Autoregressive Integrated Moving Average (ARIMA)
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
		- This is a time series forecasting algorithm
		- Not suitable for classification problems
		- Requires temporal data, which is not present in this case
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
A company is developing an ML model to predict customer churn. The model performs well on the training dataset but does not accurately predict churn for new data. 
Which solution will resolve this issue?
Options:
A. Decrease the regularization parameter to increase model complexity.
B. Increase the regularization parameter to decrease model complexity.
C. Add more features to the input data.
D. Train the model for more epochs.

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
		D. Training for more epochs would likely worsen overfitting as the model would learn the training data even more precisely
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
A company needs to train an ML model to classify images of different types of animals. The company has a large dataset of labeled images and will not label more data.  
Which type of learning should the company use to train the model?
Options:
A. Supervised learning
B. Unsupervised learning
C. Reinforcement learning
D. Active learning

Correct Answer: 
	A. Supervised learning
	--- Domain 1: Fundamentals of AI and ML, Task Statement 1.1: "Explain basic AI concepts and terminologies." ---
	Explanation: 
		Answer A is correct because:
		- The company has labeled images (labeled data)
		- The task is classification, which is a typical supervised learning problem
		- Supervised learning is used when you have input data (images) and corresponding output labels (animal types)
		- The dataset is already labeled and no more labeling is planned, making supervised learning the most appropriate approach
		The other answers are not suitable because:
		B. Unsupervised learning - Used when data is unlabeled and the goal is to find patterns/clusters
		C. Reinforcement learning - Used for learning through interaction with an environment and receiving rewards/penalties
		D. Active learning - Used when you want to selectively label more data to improve model performance, but the question states no more labeling will be done
	Resource URL:
	https://aws.amazon.com/compare/the-difference-between-machine-learning-supervised-and-unsupervised/

Q117
Which phase of the ML lifecycle determines compliance and regulatory requirements?
Options:
A. Feature engineering
B. Model training
C. Data collection
D. Business goal identification
Let me help classify and explain this question:

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
		B. Model training - This is an implementation phase that should follow already-established compliance guidelines
		C. Data collection - While compliance affects data collection, the requirements should be identified before starting data collection - Data processing phase
	Resource URL:
	https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/best-practices-by-ml-lifecycle-phase.html

Q118
A F&B sector company wants to develop an ML model to help decrease daily food waste and increase sales revenue. The company needs to continuously improve the model's accuracy.  
Which solution meets these requirements?
Options:
A. Use Amazon SageMaker and iterate with newer data.
B. Use Amazon Personalize and iterate with historical data.
C. Use Amazon CloudWatch to analyze customer orders.
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
	C. Amazon CloudWatch - This is a monitoring service, not an ML development platform
	D. Amazon Rekognition - This is specifically for image and video analysis, not for business prediction problems
	Resource URL:
	https://aws.amazon.com/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/

Q119
A company has developed an ML model to predict real estate sale prices. The company wants to deploy the model to make predictions without managing servers or infrastructure.  
  
Which solution meets these requirements?
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
A manufacturing company uses AI to inspect the quality of products and find any damages or defects.    
Which type of AI application is the company using?
Options:
A. Recommendation system
B. Natural language processing (NLP)
C. Computer vision
D. Image processing


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
		D. Image processing - While part of computer vision, it's too narrow and refers to manipulating images rather than understanding their content
	Resource URL:
		https://aws.amazon.com/what-is/computer-vision/


Q122
Some retail company wants to create an ML model to predict customer satisfaction. The company needs fully automated model tuning.  
Which AWS service meets these requirements?
Options:
A. Amazon Personalize
B. Amazon SageMaker
C. Amazon Athena
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
		C. Amazon Athena - This is a query service for analyzing data in S3, not an ML service
		D. Amazon Comprehend - This is specifically for natural language processing tasks
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html

Q127
An ecommerce company wants to improve search engine recommendations by customizing the results for each user of the company’s ecommerce platform.  
Which AWS service meets these requirements?
Options:
A. Amazon Personalize
B. Amazon Kendra
C. Amazon Rekognition
D. Amazon Transcribe


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
	D. Amazon Transcribe - For converting speech to text, not relevant for search recommendations
	Resource URL:
	https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html

Q129
A company needs to monitor the performance of its ML systems by using a highly scalable AWS service.  
Which AWS service meets these requirements?
Options:
A. Amazon CloudWatch
B. AWS CloudTrail
C. AWS Trusted Advisor
D. AWS Config

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
	D. AWS Config - For resource configuration tracking, not for performance monitoring
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html


Q132
A company is building a mobile app for users who have a visual impairment. The app must be able to hear what users say and provide voice responses.  
Which solution will meet these requirements?
Options:
A. Use a deep learning neural network to perform speech recognition.
B. Build ML models to search for patterns in numeric data.
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
		B. ML models for numeric data - Not relevant for speech processing requirements
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
A company wants to find groups for its customers based on the customers’ demographics and buying patterns.  
Which algorithm should the company use to meet this requirement?
Options :
A. K-nearest neighbors (k-NN)
B. K-means
C. Decision tree
D. Support vector machine

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
		D. Support vector machine - This is a supervised learning algorithm for classification, not suitable for finding natural groups
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html
