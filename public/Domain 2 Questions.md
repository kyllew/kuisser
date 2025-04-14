
Q12
A software company wants to use generative AI to increase developer productivity and increasing the velocity software development. The company wants to use Amazon Q Developer.  
What Amazon Q Developer capability to help the company meet these requirements?
Options : 
A. Create software snippets, reference tracking, and open source license tracking.
B. Run an application without provisioning or managing servers.
C. Enable voice commands for coding and providing natural language search.
D. Convert audio files to text documents by using ML models.

Correct Answer:
	A. Create software snippets, reference tracking, and open source license tracking.
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.2: "Understand the capabilities and limitations of generative AI for solving business problems" ---
	Explanation:
		Amazon Q Developer (formerly CodeWhisperer) helps increase developer productivity by:
		1. Creating code snippets based on comments and context
		2. Providing reference tracking for generated code
		3. Monitoring open source license compliance
		4. Offering real-time code suggestions
		5. Helping developers write code more efficiently
		Other options are incorrect because:
		- B: Describes serverless computing, not code generation
		- C: Not a feature of Amazon Q Developer
		- D: Describes speech-to-text conversion, not code generation
		Resource URL:
		https://aws.amazon.com/q/developer/

Q18
An AI engineer wants to use a foundation model (FM) to design a search application. The search application need to have capability to handle queries that have text and images.  
Which type of FM should the AI practitioner use to power the search application?
Options :
A. Multi-modal embedding model
B. Text embedding model
C. Multi-modal generation model
D. Image generation model

Correct Answer:
	A. Multi-modal embedding model
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.1: "Explain the basic concepts of generative AI" ---
	Explanation :
		Multi-modal embedding model is the correct choice because:
		1. Can process both text and image inputs simultaneously
		2. Creates unified vector representations for different types of content
		3. Specifically designed for multi-modal search applications
		4. Can understand relationships between text and images
		5. Enables searching across different data types
		Other options are incorrect because:
		- B: Text embedding model only handles text, not images
		- C: Multi-modal generation model is for creating content, not optimized for search
		- D: Image generation model only creates images, not suitable for search
		Resource URL:
		https://aws.amazon.com/what-is/embeddings-in-machine-learning/
		https://docs.aws.amazon.com/nova/latest/userguide/rag-multimodal.html

Q21
Which feature of Amazon OpenSearch Service gives companies the ability to build vector database applications?
Options :
A. Integration with Amazon S3 for object storage
B. Support for geospatial indexing and queries
C. Scalable index management and nearest neighbor search capability
D. Ability to perform real-time analysis on streaming data

Correct Answer:
	 C. Scalable index management and nearest neighbor search capability
	 --- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications" ---
	 Explanation:
		Scalable index management and nearest neighbor search capability is the correct answer because:
		1. Vector databases require efficient similarity search capabilities
		2. k-Nearest Neighbor (kNN) search is essential for vector operations
		3. Scalable indexing is crucial for vector database performance
		4. Enables efficient storage and retrieval of vector embeddings
		5. Essential for generative AI applications using embeddings
		Other options are incorrect because:
		A: S3 integration is for object storage, not vector operations
		B: Geospatial indexing is for location-based queries
		D: Real-time analysis is for streaming data, not vector operations
	Resource URL:
		https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html
		https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html

Q22
Which option is a use case for generative AI models?
Options :
A. Improving network security by using intrusion detection systems
B. Creating photorealistic images / poster from text descriptions for digital marketing
C. Enhancing database performance by using optimized indexing
D. Analyzing financial data to forecast stock market trends

Correct Answer :
	B. Creating photorealistic images from text descriptions for digital marketing
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.2: "Understand the capabilities and limitations of generative AI for solving business problems" ---
	Explanation:
		Creating photorealistic images from text descriptions is the correct use case because:
		1. It's a core capability of generative AI models (like DALL-E, Stable Diffusion)
		2. Demonstrates content generation from descriptions
		3. Shows creative application in digital marketing
		4. Utilizes text-to-image generation capabilities
		5. Represents a practical business application of generative AI
		Other options are incorrect because:
		- A: Intrusion detection is a traditional ML classification task
		- C: Database optimization is not a generative AI use case
		- D: Stock market forecasting is a predictive analytics task
		
		Resource URL:
		https://aws.amazon.com/bedrock/generative-ai/

Q32
A startup company wants to build an application by using Amazon Bedrock. The company has a limited budget and prefers flexibility without long-term commitment.  
Which Amazon Bedrock pricing model meets these requirements?
Options:
A. On-Demand
B. Model customization
C. Provisioned Throughput
D. Spot Instance

Correct Answer:
	A. On-Demand
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications" ---
	Explanation:
		On-Demand pricing model is the correct choice because:
		1. Provides pay-as-you-go pricing without upfront commitments
		2. Offers maximum flexibility for usage
		3. No long-term financial commitment required
		4. Ideal for budget-conscious projects
		5. Charges only for actual usage
		Other options are incorrect because:
		- B: Model customization is a feature, not a pricing model
		- C: Provisioned Throughput requires capacity commitment
		- D: Spot Instance is not a pricing option for Amazon Bedrock
		Resource URL:
		https://aws.amazon.com/bedrock/pricing/


Q33
Which AWS service or feature can help an AI development team quickly deploy and consume a foundation model (FM) within the team's VPC?
Options :
A. Amazon Personalize
B. Amazon SageMaker JumpStart
C. PartyRock, an Amazon Bedrock Playground
D. Amazon SageMaker endpoints

Correct Answer: 
	B. Amazon SageMaker JumpStart
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications" ---
	Explanation:
		Amazon SageMaker JumpStart is the correct solution because:
		1. Provides quick access to pre-built foundation models
		2. Enables easy deployment within a VPC
		3. Offers one-click model deployment
		4. Integrates with SageMaker's secure infrastructure
		5. Supports rapid AI application development
		Other options are incorrect because:
		- A: Amazon Personalize is for recommendation systems, not FM deployment
		- C: PartyRock is a playground for experimenting, not for VPC deployment
		- D: SageMaker endpoints are deployment targets but don't provide quick FM access
		Resource URL:
		https://aws.amazon.com/about-aws/whats-new/2022/01/amazon-sagemaker-jumpstart-custom-vpc-kms-settings/
		https://aws.amazon.com/sagemaker/jumpstart/

Q35
A company has terabytes of data in a database that the company can use for insight and business analysis. The company wants to build an AI-based application that can build a SQL query from input text that employees provide. The employees have minimal experience with technology.  
Which solution meets these requirements?
Options
A. Generative pre-trained transformers (GPT)
B. Residual neural network
C. Support vector machine
D. WaveNet

Correct Answer:
	A. Generative pre-trained transformers (GPT)
	Explanation:
		Generative pre-trained transformers (GPT) is the correct solution because:
		1. Can understand natural language and generate SQL queries
		2. Suitable for users with minimal technical experience
		3. Can handle complex language-to-SQL translations
		4. Designed for natural language understanding and generation
		5. Well-suited for text-to-code generation tasks
		Other options are incorrect because:
		- B: Residual neural networks are primarily for image processing
		- C: Support vector machines are for classification/regression, not text-to-SQL generation
		- D: WaveNet is for audio synthesis
		Resource URL:
		https://aws.amazon.com/what-is/gpt/

Q38
A company is implementing the Amazon Nova foundation model (FM) by using Amazon Bedrock. The company needs to supplement the model by using relevant data from the company's private data sources.  
Which solution will meet this requirement?
Options:
A. Use a different FM.
B. Choose a lower temperature value.
C. Create an Amazon Bedrock knowledge base.
D. Enable model invocation logging.

Correct Answer: 
	C. Create an Amazon Bedrock knowledge base.
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications" ---
	Explanation:
	Creating an Amazon Bedrock knowledge base is the correct solution because:
	1. Allows integration of private data with foundation models
	2. Enables retrieval-augmented generation (RAG)
	3. Supplements model responses with company-specific information
	4. Maintains data privacy while enhancing model capabilities
	5. Provides controlled access to private data sources
	Other options are incorrect because:
	- A: Changing FM doesn't address need for private data integration
	- B: Temperature affects output randomness, not data integration
	- D: Logging tracks usage but doesn't add private data capabilities
	Resource URL:
	https://aws.amazon.com/bedrock/knowledge-bases/

Q45
An ecommerce company wants to build a solution to determine customer sentiments based on written customer reviews of products.  
Which AWS services meet these requirements? (Choose two.)
Options :
A. Amazon Lex
B. Amazon Comprehend
C. Amazon Polly
D. Amazon Bedrock
E. Amazon Rekognition

Correct Answer: 
	B. Amazon Comprehend
	D. Amazon Bedrock
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications" ---
	Explanation:
		Amazon Comprehend and Amazon Bedrock are the correct services because:
		1. Amazon Comprehend (B):
		- Specifically designed for sentiment analysis
		- Natural language processing service
		- Pre-built models for text analysis
		- Handles customer review processing effectively
		1. Amazon Bedrock (D):
		- Provides access to foundation models
		- Can perform sentiment analysis
		- Offers customizable language models
		- Suitable for text analysis tasks
		Other options are incorrect because:
		- A: Amazon Lex is for building conversational interfaces
		- C: Amazon Polly is for text-to-speech conversion
		- E: Amazon Rekognition is for image and video analysis
	Resource URL:
	https://docs.aws.amazon.com/comprehend/latest/dg/how-sentiment.html
	https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
	https://aws.amazon.com/blogs/machine-learning/detect-sentiment-from-customer-reviews-using-amazon-comprehend/

Q50
A company is using an Amazon Bedrock base model to summarize documents for an internal use case. The company trained a custom model to improve the summarization quality.  
Which action must the company take to use the custom model through Amazon Bedrock?
Options:
A. Purchase Provisioned Throughput for the custom model.
B. Deploy the custom model in an Amazon SageMaker endpoint for real-time inference.
C. Register the model with the Amazon SageMaker Model Registry.
D. Grant access to the custom model in Amazon Bedrock.

Correct Answer:
	B. Deploy the custom model in an Amazon SageMaker endpoint for real-time inference
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation:
		Answer B is correct because:
		1. Custom models need to be deployed to a SageMaker endpoint to be accessible through Amazon Bedrock
		2. SageMaker endpoints provide the infrastructure for real-time inference
		3. This is the standard way to make custom models available for integration with Bedrock
		The other answers are not suitable/not relevant because:
		A. Purchase Provisioned Throughput:
		- Not required for custom model deployment
		- Provisioned Throughput is for managing base model capacity, not custom models
		C. Register with SageMaker Model Registry:
		- While useful for model versioning, this alone doesn't make the model accessible through Bedrock
		- Model Registry is for tracking model versions and lineage
		D. Grant access in Amazon Bedrock:
		- Access management is for base models in Bedrock
		- Custom models need to be deployed first before access can be managed
	Resource URLs:
	1. Amazon Bedrock Custom Models Documentation:
	https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
	2. SageMaker Real-time Inference:
	https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
	3. Amazon Bedrock and SageMaker Integration:
	https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization.html

Q105
A company wants to use large language models (LLMs) to produce code from natural language code comments.  
Which LLM feature meets these requirements?
Options:
A. Text summarization
B. Text generation
C. Text completion
D. Text classification

Let me help classify and explain this question.

Correct Answer:
	B. Text generation
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.1: "Explain the basic concepts of generative AI." ---
	Explanation:
		Answer B is correct because:
		- Text generation is the fundamental capability needed to create code from natural language
		- This feature allows LLMs to generate entirely new content (code) based on input prompts (comments)
		- It's specifically designed to produce coherent and contextually appropriate outputs
		- Text generation can handle the complex task of converting natural language specifications into structured code
		- This is a core generative AI capability used in code generation tools like Amazon Q Developer
		The other answers are not suitable because:
		A. Text summarization
		- Reduces longer text into shorter versions
		- Does not generate new content like code
		- Focused on condensing existing information rather than creating new content
		C. Text completion
		- While similar, text completion typically focuses on completing partial text/code
		- More suitable for autocomplete scenarios
		- Not ideal for generating complete code from scratch based on comments
		D. Text classification
		- Categorizes text into predefined classes
		- Does not generate any new content
		- Cannot convert natural language into code
	Resource URLs:
	https://aws.amazon.com/what-is/large-language-model/


Q110
A company is implementing intelligent agents to provide conversational search experiences for its customers. The company needs a database service that will support storage and queries of embeddings from a generative AI model as vectors in the database.  
Which AWS service will meet these requirements?
Options:
A. Amazon Athena
B. Amazon Aurora PostgreSQL
C. Amazon Redshift
D. Amazon EMR

Correct Answer: 
	B. Amazon Aurora PostgreSQL
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation: 
	Answer B is correct because:
	- Amazon Aurora PostgreSQL supports vector storage and similarity search through the pgvector extension
	- It's specifically designed to handle vector embeddings from generative AI models
	- Aurora PostgreSQL can efficiently perform vector similarity searches which is essential for conversational AI applications
	- It integrates well with other AWS AI services and can handle the vector database requirements for generative AI applications
	The other answers are not suitable because:
	A. Amazon Athena - It's a query service for analyzing data in S3 using SQL, but doesn't support vector storage and similarity search
	C. Amazon Redshift - While it's a data warehouse solution, it doesn't natively support vector storage and similarity search like Aurora PostgreSQL
	D. Amazon EMR - It's a big data processing service but isn't designed for vector storage and similarity search operations
	Resource URL:
	https://aws.amazon.com/about-aws/whats-new/2023/07/amazon-aurora-postgresql-pgvector-vector-storage-similarity-search/

Q112
A pharmaceutical company wants to analyze user reviews of new medications and provide a concise overview for each medication.  
Which solution meets these requirements?
Options:
A. Create a time-series forecasting model to analyze the medication reviews by using Amazon Personalize.
B. Create medication review summaries by using Amazon Bedrock large language models (LLMs).
C. Create a classification model that categorizes medications into different groups by using Amazon SageMaker.
D. Create medication review summaries by using Amazon Rekognition.

Correct Answer: 
	B. Create medication review summaries by using Amazon Bedrock large language models (LLMs).
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.2: "Understand the capabilities and limitations of generative AI for solving business problems." ---
	Explanation: 
		Answer B is correct because:
		- Amazon Bedrock's LLMs are specifically designed for natural language processing tasks like text summarization
		- LLMs can understand context and generate concise summaries from multiple reviews
		- This solution directly addresses the requirement of providing concise overviews from user reviews
		- Amazon Bedrock provides access to various foundation models suitable for text summarization tasks
		The other answers are not suitable because:
		A. Amazon Personalize - This is for recommendation systems, not text summarization
		C. Amazon SageMaker classification model - While useful for categorization, it's not designed for generating text summaries
		D. Amazon Rekognition - This is for image and video analysis, not text processing
	Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html




Q120
An insurance company wants to develop an AI application to help its employees check open customer claims, identify details for a specific claim, and access documents for a claim.  
Which solution meets these requirements?
Options:
A. Use Agents for Amazon Bedrock with Amazon Fraud Detector to build the application.
B. Use Agents for Amazon Bedrock with Amazon Bedrock knowledge bases to build the application.
C. Use Amazon Personalize with Amazon Bedrock knowledge bases to build the application.
D. Use Amazon SageMaker to build the application by training a new ML model.


Correct Answer: 
	B. Use Agents for Amazon Bedrock with Amazon Bedrock knowledge bases to build the application
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation: 
		Answer B is correct because:
		- Agents for Amazon Bedrock are designed to help build AI applications that can perform specific tasks
		- Bedrock knowledge bases allow integration of company-specific information (like customer claims and documents)
		- This combination enables:
		  * Natural language interaction with claims data
		  * Document retrieval and information extraction
		  * Integration with existing business systems
		  * Contextual responses based on company data
		The other answers are not suitable because:
		A. Amazon Fraud Detector - This is specifically for fraud detection, not for general claim processing
		C. Amazon Personalize - This is for recommendation systems, not for document retrieval and claim processing
		D. Amazon SageMaker - Training a new ML model would be unnecessary and overcomplicated when existing foundation models can be used with knowledge bases
	Resource URL:
	https://aws.amazon.com/blogs/machine-learning/automate-the-insurance-claim-lifecycle-using-amazon-bedrock-agents-and-knowledge-bases/

Q125
A company has developed a large language model (LLM) and wants to make the LLM available to multiple internal teams. The company needs to select the appropriate inference mode for each team.  
Hotspot
Requirement 1 The company's chatbot needs predictions from the LLM to understand user' intent with minimal latency
Requirement 2 A data processing job needs to query the LLM to process gigabytes of text files on weekends
Requirement 3 The company's engineering team needs to create an API that can process small pieces of text content and provide low-latency predictions

Options :
A. Requirement 1 - Batch transform; Requirement 2 - Real-time inference; Requirement 3 - Real-time inference
B. Requirement 1 - Real-time inference; Requirement 2 - Real-time inference; Requirement 3 - Real-time inference
C. Requirement 1 - Real-time inference; Requirement 2 - Batch transform; Requirement 3 - Real-time inference
D. Requirement 1 - Batch transform; Requirement 2 - Batch transform; Requirement 3 - Real-time inference

Correct Answer: 
	C. Requirement 1 - Real-time inference; Requirement 2 - Batch transform; Requirement 3 - Real-time inference
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation: 
		Answer C is correct because:
		Requirement 1 (Real-time inference):
		- Chatbot needs minimal latency for user interactions
		- Real-time inference is ideal for interactive applications
		- Provides immediate responses needed for chat applications
		Requirement 2 (Batch transform):
		- Processing large amounts of text files
		- Weekend processing indicates non-real-time needs
		- Batch transform is cost-effective for large-scale processing
		- Ideal for offline processing of gigabytes of data
		Requirement 3 (Real-time inference):
		- API needs low-latency predictions
		- Processing small pieces of text
		- Real-time inference matches API response time requirements
		The other answers are not suitable because:
		They don't properly match the inference types to the use case requirements, particularly mixing up batch and real-time needs.
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
	https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html

Q131
Which AWS service makes foundation models (FMs) available to help users build and scale generative AI applications?
Options:
A. Amazon Q Developer
B. Amazon Bedrock
C. Amazon Kendra
D. Amazon Comprehend

Correct Answer: 
	B. Amazon Bedrock
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation: 
	Answer B is correct because:
	- Amazon Bedrock is specifically designed to provide access to foundation models
	- It offers a variety of pre-trained foundation models from Amazon and third-party providers
	- Provides a unified API for accessing different foundation models
	- Enables building and scaling generative AI applications
	- Offers serverless infrastructure for deploying generative AI solutions
	The other answers are not suitable because:
	A. Amazon Q Developer - This is an AI-powered assistant for developers, not a platform for accessing foundation models
	C. Amazon Kendra - This is an enterprise search service, not for accessing foundation models
	D. Amazon Comprehend - This is for natural language processing tasks, not for accessing foundation models
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html

Q137
An ecommerce retail company is using a generative AI chatbot to respond to customer inquiries. The company wants to measure the financial effect of the chatbot on the company’s operations. 
Which metric should the company use?
Options:
A. Number of customer inquiries handled
B. Cost of training AI models
C. Cost for each customer conversation
D. Average handled time (AHT)

Correct Answer: 
	C. Cost for each customer conversation
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.2: "Understand the capabilities and limitations of generative AI for solving business problems." ---
	Explanation: 
	Answer C is correct because:
	- Cost per conversation directly measures the financial impact of the chatbot
	- It allows comparison with human agent costs
	- Helps determine ROI of the generative AI implementation
	- Provides clear financial metrics for operational impact
	- Enables business value assessment of the AI solution
	The other answers are not suitable because:
	A. Number of inquiries handled - Volume metric but doesn't show financial impact
	B. Cost of training AI models - One-time cost, doesn't reflect ongoing operational impact
	D. Average handled time - Efficiency metric but doesn't directly show financial impact
	Resource URL:
	https://aws.amazon.com/bedrock/pricing/

Q141
A company’s staffs assist the customer by giving product descriptions and recommendations when customers call the customer service center. These recommendations are given based on where the customers are located. The company wants to use foundation models (FMs) to automate this process.  
Which AWS service meets these requirements?
Options:
A. Amazon Macie
B. Amazon Transcribe
C. Amazon Bedrock
D. Amazon Textract

Let me help classify and explain this question:

Correct Answer: 
	C. Amazon Bedrock
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.3: "Describe AWS infrastructure and technologies for building generative AI applications." ---
	Explanation: 
		Answer C is correct because:
		- Amazon Bedrock provides access to foundation models for generating text
		- It can handle context-aware content generation (location-based recommendations)
		- Suitable for automating customer service tasks
		- Can generate personalized product descriptions and recommendations
		- Offers various foundation models optimized for different use cases
		The other answers are not suitable because:
		A. Amazon Macie - This is for discovering and protecting sensitive data, not for generating recommendations
		B. Amazon Transcribe - This converts speech to text, not for generating recommendations
		D. Amazon Textract - This extracts text from documents, not for generating recommendations
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html


Q147
A company deployed an AI/ML solution to help customer service agents respond to frequently asked questions. The questions can change over time. The company wants to give customer service agents the ability to ask questions and receive automatically generated answers to common customer questions.  
Which strategy will meet these requirements MOST cost-effectively?
Options:
A. Fine-tune the model regularly.
B. Train the model by using context data.
C. Pre-train and benchmark the model by using context data.
D. Use Retrieval Augmented Generation (RAG) with prompt engineering techniques.

Correct Answer: 
	D. Use Retrieval Augmented Generation (RAG) with prompt engineering techniques
	--- Domain 2: Fundamentals of Generative AI, Task Statement 2.2: "Understand the capabilities and limitations of generative AI for solving business problems." ---
	Explanation: 
	Answer D is correct because:
	- RAG allows dynamic incorporation of new information without model retraining
	- It's cost-effective as it doesn't require frequent model updates
	- Combines existing model knowledge with up-to-date context
	- Uses prompt engineering to improve response quality
	- Perfect for handling changing FAQ content
	The other answers are not suitable because:
	A. Fine-tune regularly - More expensive and time-consuming than RAG
	B. Train with context data - Requires complete retraining, not cost-effective
	C. Pre-train and benchmark - Resource-intensive and doesn't handle dynamic updates well
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
