Q1
A company wants to make forecasts each quarter to decide how to optimize operations so that it will meet expected demand. ML models is being used to make these forecasts.  An AI engineer is required to create a report about the trained ML models to provide transparency and explainability to company stakeholders. What should the AI engineer include in the report to meet the transparency and explainability requirements?
Options :
A. Code for model training
B. Partial dependence plots (PDPs)
C. Sample data for training
D. Model convergence tables

Correct Answer :
	B. Partial dependence plots (PDPs)
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Explain model transparency and explainability" ---
	Explanation:
		Partial Dependence Plots (PDPs) are the most appropriate choice for providing transparency and explainability because:
			1. PDPs show how specific features influence model predictions while holding other features constant
			2. They provide visual, interpretable representations of complex model behaviors
			3. PDPs help stakeholders understand feature relationships and their impact on forecasts
			4. They are widely accepted tools for model explainability in business contexts
		Other options are incorrect because:
			- A: Code for model training is too technical for most stakeholders and doesn't explain model decisions
			- C: Sample training data doesn't explain how the model makes decisions
			- D: Model convergence tables show technical training metrics but don't explain feature relationships or model behavior	
		Resource URL:
			https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html	
			PDPs are supported by Amazon SageMaker Clarify, which provides tools for model explainability and bias detection. This aligns with responsible AI principles of transparency and interpretability in ML systems.

Q8
A company is building a solution to generate images for protective eyewear. The solution must have high accuracy and must minimize the risk of incorrect annotations.  
Which solution will meet these requirements?
Options :
A. Human-in-the-loop validation by using Amazon SageMaker Ground Truth Plus
B. Data augmentation by using an Amazon Bedrock knowledge base
C. Image recognition by using Amazon Rekognition
D. Data summarization by using Amazon QuickSight Q

Correct Answer :
	A. Human-in-the-loop validation by using Amazon SageMaker Ground Truth Plus
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible" ---
	Explanation:
		Human-in-the-loop validation using Amazon SageMaker Ground Truth Plus is the correct solution because:
		1. Ensures high accuracy through human verification for safety equipment imaging
		2. Provides quality control and validation by human experts
		3. Minimizes risks of incorrect annotations through expert oversight
		4. Aligns with responsible AI principles for safety-critical applications
		5. Combines ML efficiency with human expertise for optimal results
		Other options are incorrect because:
		- B: Data augmentation doesn't address annotation accuracy requirements
		- C: Rekognition is for image analysis, not annotation validation
		- D: QuickSight Q is for business intelligence visualization, not relevant for this use case
		Resource URL:
			https://aws.amazon.com/sagemaker-ai/groundtruth/faqs/

Q29
An IT company wants to use a large language model (LLM) to develop a conversational agent for their contact center. The company needs to prevent the LLM from being manipulated with common prompt engineering techniques to perform undesirable actions like toxicity or expose sensitive information. Which action will reduce these risks?
Options:
A. Create a prompt template that teaches the LLM to detect attack patterns.
B. Increase the temperature parameter on invocation requests to the LLM.
C. Avoid using LLMs that are not listed in Amazon SageMaker.
D. Decrease the number of input tokens on invocations of the LLM.

Correct Answer: 
	A. Create a prompt template that teaches the LLM to detect attack patterns.
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible" ---
	Explanation:
		Creating a prompt template to detect attack patterns is the correct action because:
		1. Helps establish guardrails for model behavior
		2. Creates defensive mechanisms against prompt injection
		3. Enables the model to recognize and resist manipulation attempts
		4. Maintains system security while preserving functionality
		5. Implements responsible AI practices
		Other options are incorrect because:
		- B: Increasing temperature makes outputs more random, doesn't improve security
		- C: Model source is not relevant to preventing prompt manipulation
		- D: Reducing input tokens limits functionality without addressing security
		Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-templates-and-examples.html

Q37
An AI engineer is building a model to generate images of humans in various job professions for example as teacher, doctor. The engineer discovered that the input data is biased and that specific attributes affect the image generation and create bias in the model.  
Which technique will solve the problem?
Options :
A. Data augmentation for imbalanced classes
B. Model monitoring for class distribution
C. Retrieval Augmented Generation (RAG)
D. Watermark detection for images

Correct Answer: 
	A. Data augmentation for imbalanced classes
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible" ---
	Explanation:
		Data augmentation for imbalanced classes is the correct solution because:
		1. Helps balance representation across different groups
		2. Reduces bias in training data
		3. Creates more diverse and representative dataset
		4. Improves model fairness
		5. Addresses root cause of bias in image generation
		Other options are incorrect because:
		- B: Monitoring only detects bias, doesn't solve it
		- C: RAG is for enhancing text generation with external data
		- D: Watermark detection is for identifying image sources
		Resource URL:
		https://aws.amazon.com/what-is/data-augmentation/
	

Q39
A medical company is customizing a foundation model (FM) for diagnostic purposes. The company needs the model to be transparent and explainable to meet regulatory requirements.  
Which solution will meet these requirements?
Options:
A. Configure the security and compliance by using Amazon Inspector.
B. Generate simple metrics, reports, and examples by using Amazon SageMaker Clarify.
C. Encrypt and secure training data by using Amazon Macie.
D. Gather more data. Use Amazon Rekognition to add custom labels to the data.

Correct Answer: 
	B. Generate simple metrics, reports, and examples by using Amazon SageMaker Clarify.
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Recognize the importance of transparent and explainable models" ---
	Explanation:
	Using Amazon SageMaker Clarify is the correct solution because:
	1. Provides model explainability features
	2. Generates interpretable reports and metrics
	3. Helps meet regulatory transparency requirements
	4. Offers insights into model decision-making
	5. Supports compliance with medical regulations
	Other options are incorrect because:
	- A: Amazon Inspector is for security assessment, not model explainability
	- C: Macie is for data security, not model transparency
	- D: Adding custom labels doesn't address model explainability needs
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html
	https://aws.amazon.com/sagemaker/clarify/

Q43
Which functionality does Amazon SageMaker Clarify provide?
Options :
A. Integrates a Retrieval Augmented Generation (RAG) workflow
B. Monitors the quality of ML models in production
C. Documents critical details about ML models
D. Identifies potential bias during data preparation

Correct Answer: 
	D. Identifies potential bias during data preparation
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible" ---
	Explanation:
		Amazon SageMaker Clarify's primary functionality of identifying potential bias during data preparation is the correct answer because:
		1. Helps detect and measure bias in training data
		2. Supports responsible AI development
		3. Enables early identification of fairness issues
		4. Provides bias metrics and analysis
		5. Essential for developing unbiased AI systems
		Other options are incorrect because:
		- A: RAG workflow integration is not a Clarify feature
		- B: Model quality monitoring is primarily handled by SageMaker Model Monitor
		- C: Model documentation is handled by Model Cards, not Clarify
		Resource URL:
		https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-detect-data-bias.html
		https://aws.amazon.com/sagemaker-ai/clarify/

Q101
A payment system company is developing an ML model to make loan approvals. The company must implement a solution to detect bias in the model. The company must also be able to explain the model's predictions.  
Which solution will meet these requirements?
Options :
A. Amazon SageMaker Clarify
B. Amazon SageMaker Data Wrangler
C. Amazon SageMaker Model Cards
D. AWS AI Service Cards

Let me help classify and explain this question.

Correct Answer:
	A. Amazon SageMaker Clarify
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Recognize the importance of transparent and explainable models." ---
	Explanation:
	Answer A is correct because:
	- Amazon SageMaker Clarify helps detect potential bias in ML models and provides model explainability
	- It provides both pre-training bias metrics and post-training bias metrics
	- It generates feature importance values using SHAP (SHapley Additive exPlanations) values
	- Specifically designed for sensitive use cases like loan approvals where bias detection and model explanability are crucial requirements
	- Provides reports that help meet regulatory requirements for model transparency
	The other answers are not suitable because:
	B. Amazon SageMaker Data Wrangler
	- Primarily focused on data preparation and transformation
	- Does not provide bias detection or model explainability features
	- Used for data preprocessing, not model interpretation
	C. Amazon SageMaker Model Cards
	- Used for documenting model details and intended use cases
	- While they can include bias and fairness considerations, they don't provide active bias detection or model explanability
	D. AWS AI Service Cards
	- Documentation that provides information about AWS AI services
	- Do not provide actual bias detection or model explainability capabilities
	- Are informational resources rather than technical solutions
	Resource URLs:
	1. Amazon SageMaker Clarify Documentation:
	https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-configure-processing-jobs.html
	2. Bias Detection with SageMaker Clarify:
	https://aws.amazon.com/sagemaker-ai/clarify/

Q111
A banking solution is needed to use AI so that make loan approval decisions by using a foundation model (FM). For security and audit purposes, the company needs the AI solution's decisions to be explainable.  
Which factor relates to the explainability of the AI solution's decisions?
Options:
A. Model complexity
B. Training time
C. Number of hyperparameters
D. Deployment time
Let me help classify and explain this question:

Correct Answer: 
	A. Model complexity
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Recognize the importance of transparent and explainable models." ---
	Explanation: 
		Answer A is correct because:
		- Model complexity directly impacts the explainability of AI decisions
		- Simpler models are generally more interpretable and easier to explain
		- In financial services, where decisions need to be justified (like loan approvals), understanding how the model arrives at its decisions is crucial
		- Complex models (like deep neural networks) can be "black boxes" making it harder to explain their decisions
		- Model complexity affects the ability to provide clear reasoning for decisions, which is essential for regulatory compliance in financial services
		The other answers are not suitable because:
		B. Training time - This is a computational metric and doesn't affect the model's explainability
		C. Number of hyperparameters - While related to model complexity, it's not directly tied to explainability
		D. Deployment time - This is an operational metric and has no bearing on model explainability
	Resource URL:
	https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlper-13.html


Q113
A company wants to build a lead prioritization application for its sales team to contact potential customers. The application must give sales employees the ability to view and adjust the weights assigned to different variables in the model based on domain knowledge and expertise.  
Which ML model type meets these requirements?
Options :
A. Logistic regression model
B. Deep learning model built on principal components
C. K-nearest neighbors (k-NN) model
D. Neural network

Correct Answer: 
	A. Logistic regression model
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Recognize the importance of transparent and explainable models." ---
	Explanation: 
		Answer A is correct because:
		- Logistic regression models are inherently interpretable and transparent
		- The weights (coefficients) in logistic regression can be easily viewed and adjusted
		- Domain experts can understand the relationship between input variables and predictions
		- The model allows for direct manipulation of feature importance through weight adjustments
		- It provides a good balance between model performance and interpretability for business users
		The other answers are not suitable because:
		B. Deep learning model built on principal components - Principal components transform the original features, making it harder for domain experts to interpret and adjust
		C. K-nearest neighbors (k-NN) model - Doesn't use weights in the traditional sense and cannot be easily adjusted based on domain expertise
		D. Neural network - Complex "black box" model where weights are not easily interpretable or adjustable by business users
	Resource URL:
	https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/

Q123
Which technique can a company use to lower bias and toxicity in generative AI applications during the post-processing ML lifecycle?
Options:
A. Human-in-the-loop
B. Data augmentation
C. Feature engineering
D. Adversarial training
Let me help classify and explain this question:

Correct Answer: 
	A. Human-in-the-loop
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible." ---
	Explanation: 
	Answer A is correct because:
	- Human-in-the-loop provides direct human oversight and intervention in the AI system's output
	- Humans can review and filter out biased or toxic content before it reaches end users
	- This approach allows for continuous monitoring and improvement of AI outputs
	- Human reviewers can provide feedback to improve the system's performance and reduce bias
	- It's a proven technique for ensuring responsible AI deployment
	The other answers are not suitable because:
	B. Data augmentation - This is a pre-processing technique that doesn't address post-processing bias reduction
	C. Feature engineering - This is a pre-processing step focused on input data preparation
	D. Adversarial training - This is primarily used during model training, not post-processing
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-use-augmented-ai-a2i-human-review-loops.html
	https://aws.amazon.com/sagemaker-ai/groundtruth/faqs/

Q124
A bank has fine-tuned a large language model (LLM) to expedite the loan approval process. During an external audit of the model, the company discovered that the model was approving loans at a faster pace for a specific demographic than for other demographics.  
How should the bank fix this issue MOST cost-effectively?
Options:
A. Include more diverse training data. Fine-tune the model again by using the new data.
B. Use Retrieval Augmented Generation (RAG) with the fine-tuned model.
C. Use AWS Trusted Advisor checks to eliminate bias.
D. Pre-train a new LLM with more diverse training data.

Correct Answer: 
	A. Include more diverse training data. Fine-tune the model again by using the new data.
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible." ---
	Explanation: 
		Answer A is correct because:
		- Adding diverse training data addresses the root cause of demographic bias
		- Fine-tuning with balanced data is more cost-effective than pre-training a new model
		- This approach directly targets the bias issue while maintaining model efficiency
		- It's a practical way to improve model fairness without complete model rebuild
		- Fine-tuning is less resource-intensive than other options
		The other answers are not suitable because:
		B. RAG - While useful for adding context, it doesn't address underlying model bias in decision-making
		C. AWS Trusted Advisor - This service is for AWS infrastructure optimization, not for addressing ML model bias
		D. Pre-training a new LLM - This is extremely costly and time-consuming compared to fine-tuning
	Resource URL:
	https://aws.amazon.com/blogs/aws/customize-models-in-amazon-bedrock-with-your-own-data-using-fine-tuning-and-continued-pre-training/

Q144
A company is using a generative AI model to develop a digital assistant. The model’s responses occasionally include undesirable and potentially harmful content.  
Select the correct Amazon Bedrock filter policy from the following list for each mitigation action. Hotspot - Each filter policy should be selected one time.
Action 1 = Block input prompts or model responses that contain harmful content such as hate, insults, violence, or misconduct
Action 2 = Avoid subjects related to illegal investment advise or legal advice
Action 3 = Detect and block specific offensive term
Action 14 = Detect and filter out information in the model's responses that is not grounded in the provided source information

Filter Policy 1 = Content filters;
Filter Policy 2 = Contextual grounding check;
Filter Policy 3 = Denied topics;
Filter Policy 4 = Word filters;

Options :
A. Action 1 - Policy 1, Action 2 - Policy 2, Action 3 - Policy 3, Action 4 - Policy 4
B. Action 1 - Policy 1, Action 2 - Policy 3, Action 3 - Policy 4, Action 4 - Policy 2
C. Action 1 - Policy 2, Action 2 - Policy 3, Action 3 - Policy 1, Action 4 - Policy 4
D. Action 1 - Policy 2, Action 2 - Policy 3, Action 3 - Policy 4, Action 4 - Policy 1

Correct Answer: 
	B. Action 1 - Content filters (Policy 1), Action 2 - Denied topics (Policy 3), Action 3 - Word filters (Policy 4), Action 4 - Contextual grounding check (Policy 2)
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible." ---
	Explanation: 
		Answer B is correct because:
		Action 1 - Content filters (Policy 1):
		- Specifically designed to block harmful content
		- Handles broad categories of inappropriate content
		- Addresses hate, insults, violence, and misconduct
		Action 2 - Denied topics (Policy 3):
		- Prevents discussion of specific topics
		- Perfect for blocking specific subjects like investment/legal advice
		- Controls model's topic boundaries
		Action 3 - Word filters (Policy 4):
		- Blocks specific offensive terms
		- Handles individual word-level filtering
		- Most appropriate for term-specific blocking
		Action 4 - Contextual grounding check (Policy 2):
		- Ensures responses are grounded in provided information
		- Prevents hallucinations or ungrounded responses
		- Validates response content against source material
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html


Q145
Which option is a benefit of using Amazon SageMaker Model Cards to document AI models?
Options:
A. Providing a visually appealing summary of a mode’s capabilities.
B. Standardizing information about a model’s purpose, performance, and limitations.
C. Reducing the overall computational requirements of a model.
D. Physically storing models for archival purposes.

Correct Answer: 
	B. Standardizing information about a model's purpose, performance, and limitations
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.2: "Recognize the importance of transparent and explainable models." ---
	Explanation: 
	Answer B is correct because:
	- Model Cards provide standardized documentation of model characteristics
	- They capture crucial information about model purpose and limitations
	- Help ensure transparency in model development and deployment
	- Enable better understanding of model capabilities and constraints
	- Support responsible AI practices through clear documentation
	The other answers are not suitable because:
	A. Visual appeal - While cards may be visual, their primary purpose is documentation, not aesthetics
	C. Computational requirements - Model Cards don't affect model computation
	D. Physical storage - Model Cards are for documentation, not model storage
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html


Q148
A company built an AI-powered resume screening system. The company used a large dataset to train the model. The dataset contained resumes that were not representative of all demographics.  
Which core dimension of responsible AI does this scenario present?
Options:
A. Fairness
B. Explainability
C. Privacy and security
D. Transparency

Correct Answer: 
	A. Fairness
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible." ---
	Explanation: 
		Answer A is correct because:
		- The scenario describes a potential bias issue due to unrepresentative training data
		- Fairness in AI systems requires balanced representation across all demographics
		- Using unrepresentative data can lead to discriminatory outcomes
		- This directly relates to fair treatment of all groups in AI decision-making
		- Fairness is a key principle of responsible AI development
		The other answers are not suitable because:
		B. Explainability - This relates to understanding model decisions, not demographic representation
		C. Privacy and security - This relates to data protection, not demographic representation
		D. Transparency - This relates to openness about model operation, not fairness in representation
	Resource URL:
	https://aws.amazon.com/ai/responsible-ai/




