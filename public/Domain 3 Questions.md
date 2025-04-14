Q2
A finance institution wants to build an AI application by using large language models (LLMs). The application will read legal documents and extract important points from the documents. Which solution meets these requirements?
Options :
A. Build an automatic named entity recognition system.
B. Create a recommendation engine.
C. Develop a summarization chatbot.
D. Develop a multi-language translation system.

Correct Answer :
	C. Develop a summarization chatbot.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Apply foundation models and generative AI to common business use cases" ---
	Explanation :
		A summarization chatbot is the most appropriate solution because:
		1. LLMs excel at document understanding and summarization tasks
		2. Can extract key points from complex documents like legal texts
		3. Can present information in a conversational format
		4. Maintains context while providing relevant summaries
		Other options are incorrect because:
		- A: Named entity recognition is too narrow, focuses only on identifying specific entities rather than summarizing key points
		- B: Recommendation engines are for suggesting items/content, not document analysis
		- D: Translation system doesn't address the need to extract and summarize key points
		Resource URL:
			https://aws.amazon.com/blogs/machine-learning/automate-chatbot-for-document-and-data-retrieval-using-amazon-bedrock-agents-and-knowledge-bases/
			https://aws.amazon.com/bedrock/knowledge-bases/
		This type of application can be implemented using AWS services like Amazon Bedrock with Claude or other LLMs that are particularly good at comprehension and summarization tasks. The chatbot interface allows for interactive exploration of the extracted information.

Q5
A company is using a pre-trained large language model (LLM) to build a chatbot for product recommendations. The company requires the LLM outputs to be short and written in a specific language.  
Which solution will align the LLM response quality with the company's expectations?
Options :
A. Adjust the prompt.
B. Choose an LLM of a different size.
C. Increase the temperature.
D. Increase the Top K value.

Correct Answer :
	A. Adjust the prompt.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques" ---
	Explanation:
		Adjusting the prompt is the correct solution because:
		1. Prompts can specify desired output format and length
		2. Language preferences can be included in prompt instructions
		3. Prompt engineering is the primary method to control LLM output style
		4. No model changes or retraining required
		Other options are incorrect because:
		- B: Model size doesn't control output length or language
		- C: Increasing temperature makes outputs more random, not more specific
		- D: Increasing Top K affects token sampling diversity, not output format
		Example prompt structure:
		"Provide a brief product recommendation in [specific language]. Keep the response under [X] words."
		Resource URL:
			https://aws.amazon.com/what-is/prompt-engineering/
			https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/introduction.html


Q10
A Telco company wants to use language models to create an application for inference on edge devices. The inference must have the lowest latency possible.  
Which solution will meet these requirements?
Options :
A. Deploy optimized small language models (SLMs) on edge devices.
B. Deploy optimized large language models (LLMs) on edge devices.
C. Incorporate a centralized small language model (SLM) API for asynchronous communication with edge devices.
D. Incorporate a centralized large language model (LLM) API for asynchronous communication with edge devices.

Correct Answer :
	A. Deploy optimized small language models (SLMs) on edge devices.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Describe design considerations for applications that use foundation models" ---
	Explanation:
		Deploying optimized small language models (SLMs) on edge devices is the correct solution because:
		1. SLMs are designed for edge deployment with smaller footprint
		2. Local deployment eliminates network latency
		3. Optimized models ensure efficient resource usage
		4. Edge processing provides fastest possible inference time
		5. Small models require less computational resources
		Other options are incorrect because:
		- B: LLMs are too large for efficient edge deployment
		- C: Centralized API adds network latency
		- D: Centralized LLM with API adds both size and network latency issues
	Resource URL:
		https://aws.amazon.com/blogs/industries/opportunities-for-telecoms-with-small-language-models/

Q19
A company uses a foundation model (FM) from Amazon Bedrock for an AI search tool. The company wants to fine-tune the model to be more accurate by using the company's data.  
Which strategy will successfully fine-tune the model?
Options :
A. Provide labeled data with the prompt field and the completion field. 
B. Prepare the training dataset by creating a .txt file that contains multiple lines in .csv format.
C. Purchase Provisioned Throughput for Amazon Bedrock.
D. Train the model on journals and textbooks.

Correct Answer :
	A. Provide labeled data with the prompt field and the completion field. 
	--- Domain 3: Applications of Foundation Models, Task Statement 3.3: "Describe the training and fine-tuning process for foundation models" ---
	Explanation :
		Providing labeled data with prompt and completion fields is the correct strategy because:
		1. Fine-tuning requires structured input-output pairs
		2. Prompt-completion format is the standard for fine-tuning FMs
		3. Allows the model to learn specific patterns for the company's use case
		4. Maintains proper format for model training
		5. Enables targeted improvement in model accuracy
		Other options are incorrect because:
		- B: .txt files in .csv format is not the proper format for fine-tuning
		- C: Provisioned Throughput is for inference scaling, not fine-tuning
		- D: Generic training on journals/textbooks doesn't provide targeted learning
		Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html

Q23
A company wants to build a generative AI chatbot application by using Amazon Bedrock and needs to choose a foundation model (FM). The company wants to know how much information can fit into one prompt.  
Which consideration will inform the company's decision?
Options
A. Temperature
B. Context window
C. Batch size
D. Model size

Correct Answer:
	B. Context window
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Describe design considerations for applications that use foundation models" ---
	Explanation:
		Context window is the correct consideration because:
		1. Determines maximum length of input text (prompt) the model can process
		2. Defines how much information can be included in one prompt
		3. Varies between different foundation models
		4. Critical for determining model's capacity to handle input
		5. Directly impacts model's ability to process large amounts of text
		Other options are incorrect because:
		- A: Temperature controls randomness in output, not input capacity
		- C: Batch size relates to training/inference optimization, not prompt length
		- D: Model size relates to overall parameters, not specifically to input capacity
		Resource URL:
		https://aws.amazon.com/blogs/security/context-window-overflow-breaking-the-barrier/

Q24
A company wants to make a chatbot to help customers. The chatbot will help solve technical problems without human intervention.  
The company chose a foundation model (FM) for the chatbot. The chatbot needs to produce responses that adhere to company tone. Which solution meets these requirements?
Options :
A. Set a low limit on the number of tokens the FM can produce.
B. Use batch inferencing to process detailed responses.
C. Experiment and refine the prompt until the FM produces the desired responses.
D. Define a higher number for the temperature parameter.

Correct Answer:
	C. Experiment and refine the prompt until the FM produces the desired responses.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques" ---
	Explanation:
		Experimenting and refining the prompt is the correct solution because:
		1. Prompt engineering helps control the model's output style and tone
		2. Allows fine-tuning of responses to match company voice
		3. Enables specification of desired response format
		4. Helps maintain consistency in chatbot responses
		5. Can include examples of proper tone and style
		Other options are incorrect because:
		A: Token limit affects response length, not tone or quality
		B: Batch inferencing is for processing multiple requests, not improving response quality
		D: Higher temperature increases randomness, potentially reducing consistency
		Resource URL:
		https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-prompt-engineering.html

Q25
A company wants to use a large language model (LLM) on Amazon Bedrock for sentiment analysis. The company wants to classify the sentiment of text passages as positive or negative.  
Which prompt engineering strategy meets these requirements?
Options :
A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.
B. Provide a detailed explanation of sentiment analysis and how LLMs work in the prompt.
C. Provide the new text passage to be classified without any additional context or examples.
D. Provide the new text passage with a few examples of unrelated tasks, such as text summarization or question answering.

Correct Answer: 
	A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques" ---
	Explanation:
	Providing examples with corresponding labels (few-shot prompting) is the correct strategy because:
	1. Demonstrates the desired format and output to the model
	2. Helps model understand the specific task requirements
	3. Improves accuracy through example-based learning
	4. Provides clear context for sentiment classification
	5. Shows positive and negative examples for better understanding
	Other options are incorrect because:
	- B: Technical explanations don't improve model performance for this task
	- C: Zero-shot approach (no examples) may be less effective
	- D: Unrelated examples could confuse the model and reduce accuracy
	Resource URL:
	https://aws.amazon.com/what-is/prompt-engineering/
	https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-prompt-engineering.html
	Example prompt structure:

	Text: "This product is terrible"
	Sentiment: Negative
	
	Text: "I love this service"
	Sentiment: Positive
	
	Text: [new text to classify]
	Sentiment:



Q41
A company is training a foundation model (FM). The company wants to increase the accuracy of the model up to a specific acceptance level.  
Which solution will meet these requirements?
Options :
A. Decrease the batch size.
B. Increase the epochs.
C. Decrease the epochs.
D. Increase the temperature parameter.

Correct Answer:
	B. Increase the epochs.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.3: "Describe the training and fine-tuning process for foundation models" ---
	Explanation:
		Increasing the epochs is the correct solution because:
		1. Allows model to train longer on the dataset
		2. Gives more opportunities to learn patterns
		3. Helps improve model accuracy
		4. Enables reaching desired accuracy level
		5. Standard approach for improving model performance
		Other options are incorrect because:
		- A: Decreasing batch size may affect training stability but not necessarily improve accuracy
		- C: Decreasing epochs reduces training time, likely reducing accuracy
		- D: Temperature affects output randomness during inference, not training accuracy
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-fine-tune.html

Q42
A company is building a large language model (LLM) question answering chatbot. The company wants to decrease the number of actions call center employees need to take to respond to customer questions.  
Which business objective should the company use to evaluate the effect of the LLM chatbot?
Options:
A. Website engagement rate
B. Average call duration
C. Corporate social responsibility
D. Regulatory compliance

Correct Answer: 
	B. Average call duration
	--- Domain 3: Applications of Foundation Models, Task Statement 3.4: "Describe methods to evaluate foundation model performance" ---
	Explanation:
		Average call duration is the correct business objective because:
		1. Directly measures reduction in call center employee actions
		2. Quantifies efficiency improvement in customer service
		3. Provides measurable impact of LLM chatbot implementation
		4. Aligns with goal of reducing employee workload
		5. Clear metric for evaluating chatbot effectiveness
		Other options are incorrect because:
		- A: Website engagement doesn't measure call center efficiency
		- C: Corporate social responsibility isn't relevant to call center efficiency
		- D: Regulatory compliance doesn't measure operational improvement
	Resource URL:
		https://docs.aws.amazon.com/connect/latest/adminguide/bot-metrics.html


Q46
A company wants to use large language models (LLMs) with Amazon Bedrock to develop a chat interface for the company's product manuals. The manuals are stored as PDF files.  
Which solution meets these requirements MOST cost-effectively?
Options :
A. Use prompt engineering to add one PDF file as context to the user prompt when the prompt is submitted to Amazon Bedrock.
B. Use prompt engineering to add all the PDF files as context to the user prompt when the prompt is submitted to Amazon Bedrock.
C. Use all the PDF documents to fine-tune a model with Amazon Bedrock. Use the fine-tuned model to process user prompts.
D. Upload PDF documents to an Amazon Bedrock knowledge base. Use the knowledge base to provide context when users submit prompts to Amazon Bedrock.

Correct Answer: 
	D. Upload PDF documents to an Amazon Bedrock knowledge base. Use the knowledge base to provide context when users submit prompts to Amazon Bedrock.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Describe design considerations for applications that use foundation models" ---
	Explanation:
	Using an Amazon Bedrock knowledge base is the most cost-effective solution because:
	1. Efficiently stores and indexes PDF content
	2. Only retrieves relevant context when needed
	3. Optimizes token usage in prompts
	4. Avoids expensive fine-tuning process
	5. Provides scalable document management
	Other options are incorrect because:
	- A: Single PDF context limits knowledge access
	- B: Including all PDFs in every prompt wastes tokens and increases costs
	- C: Fine-tuning is expensive and unnecessary for this use case
	Resource URL:
	https://aws.amazon.com/bedrock/knowledge-bases/

Q47
A social media company wants to use a large language model (LLM) for content moderation. The company wants to evaluate the LLM outputs for bias and potential discrimination against specific groups or individuals.  
Which data source should the company use to evaluate the LLM outputs with the LEAST administrative effort?
Options:
A. User-generated content
B. Moderation logs
C. Content moderation guidelines
D. Benchmark datasets
Let me help classify and explain this question.

Correct Answer: 
	D. Benchmark datasets
	--- Domain 3: Applications of Foundation Models, Task Statement 3.4: "Describe methods to evaluate foundation model performance." ---
	Explanation: 
		Answer D is correct because:
		- Benchmark datasets are standardized, pre-existing collections of data specifically designed to evaluate model performance
		- They require the least administrative effort as they are:
		  - Already cleaned and formatted
		  - Have known ground truth labels
		  - Are commonly used in the industry for model evaluation
		  - Don't require additional data collection or processing
		  - Can help identify biases and discrimination patterns systematically
		The other answers are not suitable because:
		A. User-generated content
		- Requires significant effort to collect, clean, and label
		- May contain sensitive or personal information
		- Needs extensive preprocessing and validation
		B. Moderation logs
		- Requires historical data collection
		- May be incomplete or inconsistent
		- Needs significant processing and standardization
		- May contain sensitive information
		C. Content moderation guidelines
		- Are static rules rather than actual test data
		- Don't provide concrete examples for evaluation
		- Would need to be converted into test cases, requiring additional effort
	Resource URLs:
	1. AWS Documentation on Model Evaluation:
	https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
	
	2. AWS Responsible AI Resources:
	https://aws.amazon.com/machine-learning/responsible-ai/

Q48
A company wants to use a pre-trained generative AI model to generate content for its marketing campaigns. The company needs to ensure that the generated content aligns with the company's brand voice and messaging requirements.  
Which solution meets these requirements?
Options :
A. Optimize the model's architecture and hyperparameters to improve the model's overall performance.
B. Increase the model's complexity by adding more layers to the model's architecture.
C. Create effective prompts that provide clear instructions and context to guide the model's generation.
D. Select a large, diverse dataset to pre-train a new generative model.

Correct Answer:
	C. Create effective prompts that provide clear instructions and context to guide the model's generation
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation:
		Answer C is correct because:
		1. Prompt engineering is the most efficient and practical way to control foundation model outputs
		2. Well-crafted prompts can include specific instructions about brand voice and tone
		3. Using prompts requires no model modification and is cost-effective
		4. It aligns with best practices for using pre-trained foundation models
		The other answers are not suitable/not relevant because:
		A. Optimizing model architecture and hyperparameters:
		- Unnecessarily complex for a pre-trained model
		- Doesn't specifically address brand voice requirements
		- Time-consuming and resource-intensive
		B. Increasing model complexity:
		- Adding layers doesn't guarantee better alignment with brand voice
		- Could potentially make the model less stable
		- Unnecessary modification of a pre-trained model
		D. Selecting new dataset for pre-training:
		- Extremely resource-intensive and costly
		- Unnecessary when existing models can be guided through prompts
		- Doesn't guarantee better alignment with brand requirements
	Resource URLs:
	1. AWS Bedrock Prompt Engineering Guide:
	https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering.html
	2. AWS Foundation Models Overview:
	https://aws.amazon.com/bedrock/foundation-models/
	3. AWS Bedrock Best Practices:
	https://docs.aws.amazon.com/bedrock/latest/userguide/best-practices.html

Q49
A loan company is building a generative AI-based solution to offer new applicants discounts based on specific business criteria. The company wants to build and use an AI model responsibly to minimize bias that could negatively affect some customers.  
Which actions should the company take to meet these requirements? (Choose two.)
Options:
A. Detect imbalances or disparities in the data.
B. Ensure that the model runs frequently.
C. Evaluate the model's behavior so that the company can provide transparency to stakeholders.
D. Use the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) technique to ensure that the model is 100% accurate.
E. Ensure that the model's inference time is within the accepted limits.

Correct Answer:
	A. Detect imbalances or disparities in the data.
	C. Evaluate the model's behavior so that the company can provide transparency to stakeholders.
	--- Domain 4: Guidelines for Responsible AI, Task Statement 4.1: "Explain the development of AI systems that are responsible." ---
	Explanation:
	Answers A and C are correct because:
	1. Answer A (Detect imbalances or disparities in the data):
	- Essential for identifying and mitigating potential bias in training data
	- Helps ensure fair treatment across different customer segments
	- Key component of responsible AI development
	2. Answer C (Evaluate model's behavior for transparency):
	- Enables stakeholder understanding of model decisions
	- Supports accountability in AI systems
	- Essential for maintaining trust and compliance in financial services
	The other answers are not suitable/not relevant because:
	B. Ensuring frequent model runs:
	- Doesn't address bias or responsible AI concerns
	- Model frequency isn't related to fairness or transparency
	D. Using ROUGE technique:
	- ROUGE is primarily for text summarization evaluation
	- Not appropriate for bias detection or responsible AI implementation
	- 100% accuracy is typically not achievable and shouldn't be the primary goal
	E. Model inference time:
	- Performance metric unrelated to responsible AI or bias mitigation
	- Doesn't address fairness or transparency requirements
	Resource URLs:
	1. AWS Responsible AI Guidelines:
	https://aws.amazon.com/machine-learning/responsible-ai/
	2. AWS AI Fairness and Bias Documentation:
	https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html
	3. Amazon SageMaker Clarify:
	https://aws.amazon.com/sagemaker/clarify/

Q102
A company has developed a generative text summarization model by using Amazon Bedrock. The company will use Amazon Bedrock automatic model evaluation capabilities.  
Which metric should the company use to evaluate the accuracy of the model?
Options:
A. Area Under the ROC Curve (AUC) score
B. F1 score
C. BERTScore
D. Real world knowledge (RWK) score

Correct Answer:
	C. BERTScore
	--- Domain 3: Applications of Foundation Models, Task Statement 3.4: "Describe methods to evaluate foundation model performance." ---
	Explanation:
	Answer C is correct because:
	- BERTScore is specifically designed to evaluate text generation quality, including summarization tasks
	- It uses BERT embeddings to compute similarity between generated text and reference text
	- It captures semantic meaning better than traditional metrics
	- It's particularly effective for evaluating generative AI outputs
	- Amazon Bedrock supports BERTScore as one of its evaluation metrics for text generation tasks
	The other answers are not suitable because:
	A. Area Under the ROC Curve (AUC) score
	- Primarily used for binary classification problems
	- Not suitable for text generation or summarization tasks
	- Measures true positive vs. false positive rates, which isn't applicable to text generation
	B. F1 score
	- Mainly used for classification tasks
	- Combines precision and recall
	- Not appropriate for evaluating text generation quality
	D. Real world knowledge (RWK) score
	- This is not a standard metric for model evaluation
	- Not supported by Amazon Bedrock's evaluation capabilities
	Resource URLs:
		https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-report-programmatic.html

Q106
An international worldwide company is introducing a mobile app that helps users learn foreign languages. The app makes text more coherent by calling a large language model (LLM). The company collected a diverse dataset of text and supplemented the dataset with examples of more readable versions. The company wants the LLM output to resemble the provided examples.  
Which metric should the company use to assess whether the LLM meets these requirements?
Options:
A. Value of the loss function
B. Semantic robustness
C. Recall-Oriented Understudy for Gisting Evaluation (ROUGE) score
D. Latency of the text generation

Correct Answer: 
	C. Recall-Oriented Understudy for Gisting Evaluation (ROUGE) score
	--- Domain 3: Applications of Foundation Models, Task Statement 3.4: "Describe methods to evaluate foundation model performance." ---
	Explanation: 
		Answer C is correct because:
		- ROUGE score is specifically designed to evaluate the quality of machine-generated text by comparing it with human-written reference texts
		- In this case, the company wants to compare the LLM output with their provided examples of more readable versions
		- ROUGE measures the overlap between the generated text and reference texts, making it ideal for assessing whether the LLM output matches the desired style and readability level
		- ROUGE is particularly useful for evaluating text summarization and generation tasks, which aligns with the language learning app's requirements
		The other answers are not suitable because:
		A. Value of the loss function - This is an internal training metric and doesn't directly measure the similarity between output and reference texts
		B. Semantic robustness - This measures the model's resistance to adversarial attacks rather than text quality
		D. Latency of text generation - This measures performance speed, not text quality or similarity to reference examples
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-llms-finetuning-metrics.html

Q107
A company notices that its foundation model (FM) generates images that are unrelated to the prompts. The company wants to modify the prompt techniques to decrease unrelated images.  
Which solution meets these requirements?
Options:
A. Use zero-shot prompts.
B. Use negative prompts.
C. Use positive prompts.
D. Use ambiguous prompts.

Correct Answer: 
	B. Use negative prompts
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
		Answer B is correct because:
		- Negative prompts allow users to explicitly specify what they don't want to see in the generated images
		- By using negative prompts, the company can instruct the model to avoid generating unrelated or unwanted elements in the images
		- Negative prompting is a proven technique to improve the relevance and quality of image generation by establishing clear boundaries for what should not be included
		The other answers are not suitable because:
		A. Zero-shot prompts - These are used when no examples are provided, but won't specifically help reduce unrelated content
		C. Positive prompts - While useful, positive prompts alone don't explicitly help prevent unwanted elements
		D. Ambiguous prompts - These would actually increase the likelihood of generating unrelated images by providing unclear instructions
	Resource URL:
	https://docs.aws.amazon.com/nova/latest/userguide/prompting-image-negative.html

Q108
A company wants to use a large language model (LLM) to generate concise, feature-specific descriptions for the company’s products.  
Which prompt engineering technique meets these requirements?
Options
A. Create one prompt that covers all products. Edit the responses to make the responses more specific, concise, and tailored to each product.
B. Create prompts for each product category that highlight the key features. Include the desired output format and length for each prompt response.
C. Include a diverse range of product features in each prompt to generate creative and unique descriptions.
D. Provide detailed, product-specific prompts to ensure precise and customized descriptions.

Correct Answer: 
	B. Create prompts for each product category that highlight the key features. Include the desired output format and length for each prompt response.
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
		Answer B is correct because:
		- It follows best practices for prompt engineering by providing clear structure and expectations
		- Organizing prompts by product category ensures consistency within similar product types
		- Specifying output format and length helps control the LLM's response to meet the requirement of "concise" descriptions
		- Including key features in the prompts helps focus the model on the most relevant product attributes
		- This approach balances efficiency with specificity
		The other answers are not suitable because:
		A. Using one prompt for all products would be too generic and require additional manual editing, which is inefficient
		C. Including diverse range of features could lead to verbose and unfocused descriptions, contrary to the requirement for conciseness
		D. While detailed prompts are good, focusing solely on individual products without categorization could be inefficient and lead to inconsistent results across similar products
	Resource URL:
	https://aws.amazon.com/what-is/prompt-engineering/
	https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/

Q115
Which strategy will determine if a foundation model (FM) effectively meets business objectives?
Options:
A. Evaluate the model's performance on benchmark datasets.
B. Analyze the model's architecture and hyperparameters.
C. Assess the model's alignment with specific use cases.
D. Measure the computational resources required for model deployment.

Correct Answer: 
	C. Assess the model's alignment with specific use cases
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Describe design considerations for applications that use foundation models" ---
	Explanation: 
	Answer C is correct because:
	- A model's effectiveness is best determined by how well it performs on the specific business tasks it's meant to solve
	- Alignment with use cases directly connects to business objectives
	- This approach ensures the model is practical and valuable for the intended business purpose
	- It focuses on real-world application rather than theoretical performance
	The other answers are not suitable because:
	A. Evaluating benchmark datasets - While useful for general performance, benchmarks may not reflect specific business needs
	B. Analyzing architecture and hyperparameters - Technical details don't directly indicate business effectiveness
	D. Measuring computational resources - Operational costs are important but don't determine business effectiveness
	Resource URL:
	https://aws.amazon.com/blogs/aws/evaluate-compare-and-select-the-best-foundation-models-for-your-use-case-in-amazon-bedrock-preview/


Q130
An AI engineer is developing a prompt for an Amazon Nova model. The model is hosted on Amazon Bedrock. The AI engineer is using the model to solve numerical reasoning challenges. The AI practitioner adds the following phrase to the end of the prompt: “Ask the model to show its work by explaining its reasoning step by step.”  
  
Which prompt engineering technique is the AI practitioner using?
Options
A. Chain-of-thought prompting
B. Prompt injection
C. Few-shot prompting
D. Prompt templating
Let me help classify and explain this question:

Correct Answer: 
	A. Chain-of-thought prompting
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
		Answer A is correct because:
		- Chain-of-thought prompting explicitly asks the model to show its reasoning process step by step
		- This technique is particularly useful for numerical reasoning tasks
		- The prompt specifically requests step-by-step explanation of reasoning
		- It helps make the model's thought process transparent and verifiable
		- Chain-of-thought is proven effective for complex reasoning tasks
		The other answers are not suitable because:
		B. Prompt injection - This is a security concern where unwanted instructions are inserted into prompts
		C. Few-shot prompting - This involves providing examples to guide the model's responses
		D. Prompt templating - This refers to creating standardized prompt structures
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-prompt-engineering.html

Q133
A company wants to enhance response quality for a large language model (LLM) for complex problem-solving tasks. The tasks require detailed reasoning and a step-by-step explanation process.  
Which prompt engineering technique meets these requirements?
Options:
A. Few-shot prompting
B. Zero-shot prompting
C. Directional stimulus prompting
D. Chain-of-thought prompting

Correct Answer: 
	D. Chain-of-thought prompting
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
	Answer D is correct because:
	- Chain-of-thought prompting encourages the model to break down complex problems into steps
	- It improves reasoning quality by making the thought process explicit
	- Perfect for complex problem-solving tasks requiring detailed explanations
	- Helps in understanding and verifying the model's reasoning process
	- Enhances response quality through structured thinking
	The other answers are not suitable because:
	A. Few-shot prompting - While useful for learning from examples, doesn't specifically address step-by-step reasoning
	B. Zero-shot prompting - Doesn't provide structure for complex reasoning tasks
	C. Directional stimulus prompting - Not a standard prompt engineering technique for complex reasoning
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering.html

Q134
A company wants to keep its foundation model (FM) relevant by using the most recent data. The company wants to implement a model training strategy that includes regular updates to the FM.  
Which solution meets these requirements?
Options:
A. Batch learning
B. Continuous pre-training
C. Static training
D. Latent training


Correct Answer: 
	B. Continuous pre-training
	--- Domain 3: Applications of Foundation Models, Task Statement 3.3: "Describe the training and fine-tuning process for foundation models." ---
	Explanation: 
		Answer B is correct because:
		- Continuous pre-training allows regular updates to the model with new data
		- It helps maintain model relevance by incorporating recent information
		- Enables ongoing model improvement without complete retraining
		- Supports incremental learning from new data
		- Perfect for keeping foundation models up-to-date
		The other answers are not suitable because:
		A. Batch learning - One-time training with a fixed dataset, doesn't support regular updates
		C. Static training - Implies fixed, unchanging training, opposite of what's needed
		D. Latent training - Not a real training strategy for foundation models
		Resource URL:
		https://aws.amazon.com/about-aws/whats-new/2023/11/continued-pre-training-amazon-bedrock-preview/
		https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html

Q139
A company’s large language model (LLM) is experiencing hallucinations.  
How can the company decrease hallucinations?
Options:
A. Set up Agents for Amazon Bedrock to supervise the model training.
B. Use data pre-processing and remove any data that causes hallucinations.
C. Decrease the temperature inference parameter for the model.
D. Use a foundation model (FM) that is trained to not hallucinate.

Correct Answer: 
	C. Decrease the temperature inference parameter for the model
	--- Domain 3: Applications of Foundation Models, Task Statement 3.1: "Describe design considerations for applications that use foundation models" ---
	Explanation: 
	Answer C is correct because:
	- Temperature is a key parameter that controls the randomness in model outputs
	- Lower temperature makes the model's outputs more deterministic and focused
	- Reducing temperature helps the model stick to more probable/factual responses
	- This approach directly addresses hallucination by making responses more conservative
	- It's a practical and immediate solution for reducing hallucinations
	The other answers are not suitable because:
	A. Agents for Amazon Bedrock - Agents are for task orchestration, not for controlling hallucinations
	B. Remove data causing hallucinations - Hallucinations aren't caused by specific data points that can be removed
	D. Using a non-hallucinating FM - No foundation model is completely immune to hallucinations
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/inference-parameters.html

Q140
A company is using a large language model (LLM) on Amazon Bedrock to build a chatbot. The chatbot processes customer support requests. To resolve a request, the customer and the chatbot must interact a few times.  
Which solution gives the LLM the ability to use content from previous customer messages?
Options:
A. Turn on model invocation logging to collect messages.
B. Add messages to the model prompt.
C. Use Amazon Personalize to save conversation history.
D. Use Provisioned Throughput for the LLM.

Correct Answer: 
	B. Add messages to the model prompt
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
	Answer B is correct because:
	- Including previous messages in the prompt provides conversation context
	- This technique allows the LLM to maintain conversation coherence
	- It enables the model to reference earlier parts of the conversation
	- This is a standard prompt engineering technique for maintaining conversation history
	- Helps create more contextually appropriate responses
	The other answers are not suitable because:
	A. Model invocation logging - This is for monitoring, not for maintaining conversation context
	C. Amazon Personalize - This is for recommendations, not for conversation management
	D. Provisioned Throughput - This relates to performance capacity, not conversation context
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html
	https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management-create.html


Q143
A company is training its employees on how to structure prompts for foundation models.  
Hotspot
Select the correct prompt engineering technique from the following list for each prompt template. Each prompt engineering technique should be selected one time.
Prompt 1 : "Classify the following text as either sports, politics, or entertainment : [input text]."
Prompt 2 : "A [image 1], [image 2], and [image 3] are examples of [target class]. Classify the following image as [target class]"
Prompt 3 : "[Question.]   [Instructions to follow.]. Think step by step and walk me through your thinking process"

Options:
A. Prompt 1 : Zero-shot learning, Prompt 2 : Chain-of-thought reasoning, Prompt 3 : Few-shot learning.
B. Prompt 1 : Zero-shot learning, Prompt 2 : Few-shot learning, Prompt 3 : Chain-of-thought reasoning.
C. Prompt 1 : Few-shot learning, Prompt 2 : Zero-shot learning, Prompt 3 : Chain-of-thought reasoning.
D. Prompt 1 : Few-shot learning, Prompt 2 : Chain-of-thought reasoning, Prompt 3 : Zero-shot learning.

Correct Answer: 
	B. Prompt 1: Zero-shot learning, Prompt 2: Few-shot learning, Prompt 3: Chain-of-thought reasoning
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
	Answer B is correct because:
	Prompt 1 (Zero-shot learning):
	- Directly asks for classification without examples
	- No prior examples or demonstrations provided
	- Model must perform task based on instructions alone
	Prompt 2 (Few-shot learning):
	- Provides examples ([image 1], [image 2], [image 3])
	- Shows model examples before asking for classification
	- Uses demonstration to guide the model's response
	Prompt 3 (Chain-of-thought reasoning):
	- Explicitly asks for step-by-step thinking
	- Requires showing the reasoning process
	- Focuses on explaining the thought process
	The other answers are not suitable because they incorrectly match the prompting techniques with their corresponding examples.
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-prompt-engineering.html


Q146
What does an F1 score measure in the context of foundation model (FM) performance?
Options:
A. Model precision and recall
B. Model speed in generating responses
C. Financial cost of operating the model
D. Energy efficiency of the model’s computations

Correct Answer: 
	A. Model precision and recall
	--- Domain 3: Applications of Foundation Models, Task Statement 3.4: "Describe methods to evaluate foundation model performance." ---
	Explanation: 
	Answer A is correct because:
	- F1 score is a harmonic mean of precision and recall
	- It combines both precision (accuracy of positive predictions) and recall (ability to find all positive cases)
	- Provides a balanced measure of model performance
	- Particularly useful when dataset classes are imbalanced
	- Common metric for evaluating foundation model performance
	The other answers are not suitable because:
	B. Model speed - This is a computational metric, not what F1 score measures
	C. Financial cost - This is an operational metric, not related to F1 score
	D. Energy efficiency - This is a resource utilization metric, not what F1 score measures
	Resource URL:
	https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-evaluation-metrics.html


Q150
A company wants to improve the accuracy of the responses from a generative AI application. The application uses a foundation model (FM) on Amazon Bedrock.  
Which solution meets these requirements MOST cost-effectively?
Options:
A. Fine-tune the FM.
B. Retrain the FM.
C. Train a new FM.
D. Use prompt engineering.

Correct Answer: 
	D. Use prompt engineering
	--- Domain 3: Applications of Foundation Models, Task Statement 3.2: "Choose effective prompt engineering techniques." ---
	Explanation: 
	Answer D is correct because:
	- Prompt engineering is the most cost-effective way to improve model responses
	- Requires no model training or modification
	- Can be implemented immediately without additional infrastructure
	- Allows for iterative improvements without significant costs
	- Most efficient way to optimize model output
	The other answers are not suitable because:
	A. Fine-tune the FM - More expensive and resource-intensive than prompt engineering
	B. Retrain the FM - Very expensive and time-consuming
	C. Train a new FM - Most expensive and resource-intensive option
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html



