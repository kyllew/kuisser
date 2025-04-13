Q9
A company wants to create a chatbot by using a foundation model (FM) on Amazon Bedrock. The FM needs to access encrypted data that is stored in an Amazon S3 bucket. The data is encrypted with Amazon S3 managed keys (SSE-S3).  The FM encounters a failure when attempting to access the S3 bucket data.  
Which solution will meet these requirements?
Options :
A. Ensure that the role that Amazon Bedrock assumes has permission to decrypt data with the correct encryption key.
B. Set the access permissions for the S3 buckets to allow public access to enable access over the internet.
C. Use prompt engineering techniques to tell the model to look for information in Amazon S3.
D. Ensure that the S3 data does not contain sensitive information.

Correct Answer :
	A. Ensure that the role that Amazon Bedrock assumes has permission to decrypt data with the correct encryption key.
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems" ---
	Explanation:
		Ensuring proper IAM role permissions for decryption is the correct solution because:
		1. Amazon Bedrock needs appropriate permissions to access encrypted S3 data
		2. The role must have permissions to both access S3 and decrypt SSE-S3 encrypted data
		3. This follows AWS security best practices for secure data access
		4. Maintains data security while enabling necessary FM functionality
		Other options are incorrect because:
		- B: Making buckets public is a security risk and violates best practices
		- C: Prompt engineering doesn't resolve permission issues
		- D: Data sensitivity isn't relevant to the access permission problem
	Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html


Q13
A healthcare sector company is using Amazon Bedrock to develop an AI application. The application is hosted in a VPC. To meet regulatory compliance standards, the VPC is not allowed access to any internet traffic.  
Which AWS service or feature will meet these requirements?
Options :
A. AWS PrivateLink
B. Amazon Macie
C. Amazon CloudFront
D. Internet gateway

Correct Answer :
	A. AWS PrivateLink
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.2: "Recognize governance and compliance regulations for AI systems" ---
		Explanation:
		AWS PrivateLink is the correct solution because:
		1. Enables private connectivity to AWS services without internet access
		2. Allows secure access to Amazon Bedrock from within a VPC
		3. Meets regulatory compliance requirements for no internet access
		4. Provides private endpoint functionality
		5. Maintains network isolation while accessing AWS services
		Other options are incorrect because:
		- B: Macie is for data security and privacy, not network connectivity
		- C: CloudFront is a content delivery network requiring internet access
		- D: Internet gateway provides internet access, violating the requirement
		Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-interface-endpoints.html

Q26
A security company is using Amazon Bedrock to run foundation models (FMs). The company wants to ensure that only authorized users invoke the models. The company needs to identify any unauthorized access attempts to set appropriate AWS Identity and Access Management (IAM) policies and roles for future iterations of the FMs.  
Which AWS service should the company use to identify unauthorized users that are trying to access Amazon Bedrock?
Options :
A. AWS Audit Manager
B. AWS CloudTrail
C. Amazon Fraud Detector
D. AWS Trusted Advisor

Correct Answer: 
	B. AWS CloudTrail
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems" ---
	Explanation:
	AWS CloudTrail is the correct solution because:
	1. Records API calls for AWS services, including Amazon Bedrock
	2. Provides detailed logs of user and resource activity
	3. Helps identify unauthorized access attempts
	4. Supports security analysis and troubleshooting
	5. Enables IAM policy refinement based on actual usage patterns
	Other options are incorrect because:
	A: AWS Audit Manager is for assessing compliance programs, not real-time monitoring
	C: Amazon Fraud Detector is for detecting fraudulent activities in business transactions
	D: AWS Trusted Advisor provides best practice recommendations, not detailed access logs
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/logging-using-cloudtrail.html

Q28
An AI company schedule the evaluation of its systems and processes with the help of independent software vendors (ISVs). The company needs to receive email message notifications when an ISV's compliance reports become available.  
Which AWS service can the company use to meet this requirement?
Options:
A. AWS Audit Manager
B. AWS Artifact
C. AWS Trusted Advisor
D. AWS Data Exchange

Correct Answer:
	B. AWS Artifact
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.2: "Recognize governance and compliance regulations for AI systems" ---
	Explanation:
		AWS Artifact is the correct solution because:
		1. Provides on-demand access to AWS compliance reports
		2. Offers compliance report notifications
		3. Supports third-party auditor reports
		4. Manages compliance documentation
		5. Enables automated notification system for new reports
		Other options are incorrect because:
		A: AWS Audit Manager is for creating and managing audit reports, not accessing ISV compliance reports
		C: AWS Trusted Advisor provides best practice recommendations
		D: AWS Data Exchange is for sharing data sets, not compliance reports
		Resource URL:
		https://docs.aws.amazon.com/artifact/latest/ug/what-is-aws-artifact.html

Q30
A company is using the Generative AI Security Scoping Matrix to assess security responsibilities for its solutions. The company has identified four different solution scopes based on the matrix.  
Which solution scope gives the company the MOST ownership of security responsibilities?
Options :
A. Using a third-party enterprise application that has embedded generative AI features.
B. Building an application by using an existing third-party generative AI foundation model (FM).
C. Refining an existing third-party generative AI foundation model (FM) by fine-tuning the model by using data specific to the business.
D. Building and training a generative AI model from scratch by using specific data that a customer owns.

Correct Answer: 
D. Building and training a generative AI model from scratch by using specific data that a customer owns.
--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems" ---

Explanation:
	Building and training a generative AI model from scratch gives the most security responsibility because:
	1. Complete control over model architecture and training
	2. Full responsibility for data security
	3. End-to-end ownership of model development
	4. Maximum control over security implementations
	5. Total responsibility for model governance
	Other options have less security responsibility because:
	- A: Security mostly handled by third-party application provider
	- B: Base model security handled by third-party provider
	- C: Partial responsibility (only for fine-tuning data and process)
	Resource URL:
	https://aws.amazon.com/ai/generative-ai/security/scoping-matrix/


Q34
How can companies use large language models (LLMs) securely on Amazon Bedrock?
Options :
A. Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access.
B. Enable AWS Audit Manager for automatic model evaluation jobs.
C. Enable Amazon Bedrock automatic model evaluation jobs.
D. Use Amazon CloudWatch Logs to make models explainable and to monitor for bias.

Correct Answer: 
	A. Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access.
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems" ---
	Explanation:
		Designing clear prompts and configuring IAM with least privilege is the correct approach because:
		1. Proper IAM configuration ensures controlled access to resources
		2. Least privilege principle minimizes security risks
		3. Clear prompts help prevent prompt injection attacks
		4. Combines both security controls and proper usage
		5. Follows AWS security best practices
		Other options are incorrect because:
		- B: Audit Manager is for compliance assessment, not model security
		- C: Bedrock doesn't have automatic model evaluation jobs
		- D: CloudWatch Logs is for monitoring, not model explainability or bias detection
		Resource URL:
		https://docs.aws.amazon.com/bedrock/latest/userguide/security.html
		https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html
		https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-injection.html


Q40
A company wants to deploy a conversational chatbot to answer customer questions. The chatbot is based on a fine-tuned Amazon SageMaker JumpStart model. Because of strong restriction in the country, the application must comply with multiple regulatory frameworks.  
Which capabilities can the company show compliance for? (Choose two.)
Options
A. Auto scaling inference endpoints
B. Threat detection
C. Data protection
D. Cost optimization
E. Loosely coupled microservices

Correct Answer: 
	B. Threat detection
	C. Data protection
	Explanation: 
		Threat detection and Data protection are the correct capabilities because:
		1. Threat Detection (B):
		- Essential for security compliance
		- Monitors and detects unauthorized access
		- Ensures system security
		- Part of regulatory requirements
		2. Data Protection (C):
		- Critical for regulatory compliance
		- Ensures data privacy and security
		- Protects sensitive customer information
		- Meets data handling regulations
		Other options are incorrect because:
		- A: Auto scaling is an operational feature, not a compliance capability
		- D: Cost optimization is a business consideration, not a compliance requirement
		- E: Microservices architecture is not directly related to compliance
	Resource URL: 
		https://docs.aws.amazon.com/sagemaker/latest/dg/data-protection.html

Q104
A company is using custom models in Amazon Bedrock for a generative AI application. The company wants to use a company managed encryption key to encrypt the model artifacts that the model customization jobs create.  
Which AWS service meets these requirements?
Options:
A. AWS Key Management Service (AWS KMS)
B. Amazon Inspector
C. Amazon Macie
D. AWS Secrets Manager

Correct Answer:
	A. AWS Key Management Service (AWS KMS)
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems." ---
	Explanation:
		Answer A is correct because:
		- AWS KMS is specifically designed for creating and managing encryption keys
		- It integrates natively with Amazon Bedrock for model artifact encryption
		- Allows companies to manage their own customer master keys (CMKs)
		- Provides centralized key control and auditing capabilities
		- Supports encryption of model artifacts during customization jobs
		- Enables compliance with data security requirements
		The other answers are not suitable because:
		B. Amazon Inspector
		- Security assessment service for AWS resources
		- Focused on vulnerability and compliance assessments
		- Does not provide encryption key management capabilities
		C. Amazon Macie
		- Data security and privacy service that uses machine learning
		- Discovers and protects sensitive data
		- Does not provide encryption key management
		D. AWS Secrets Manager
		- Manages secrets like database credentials and API keys
		- Not designed for encryption key management
		- Cannot be used to encrypt model artifacts directly
	Resource URLs:
		https://docs.aws.amazon.com/bedrock/latest/userguide/data-encryption.html

Q126
A company needs to log all requests made to its Amazon Bedrock API. The company must retain the logs securely for 5 years at the lowest possible cost.  
Which combination of AWS service and storage class meets these requirements? (Choose two.)
Options
A. AWS CloudTrail
B. Amazon CloudWatch
C. AWS Audit Manager
D. Amazon S3 Intelligent-Tiering
E. Amazon S3 Standard

Correct Answer: 
A. AWS CloudTrail
D. Amazon S3 Intelligent-Tiering
--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.1: "Explain methods to secure AI systems." ---

Explanation: 
	Answer A (AWS CloudTrail) is correct because:
	- CloudTrail is specifically designed for logging API activity
	- It automatically logs all API calls made to AWS services, including Bedrock
	- Provides detailed audit trails needed for security and compliance
	- Integrates seamlessly with S3 for log storage
	Answer D (Amazon S3 Intelligent-Tiering) is correct because:
	- Automatically moves data between access tiers based on usage patterns
	- Optimizes costs for long-term storage (5 years requirement)
	- Provides the lowest possible cost for long-term storage with varying access patterns
	- Maintains high durability and security for log data
	The other answers are not suitable because:
	B. Amazon CloudWatch - Primarily for monitoring and operational data, not API activity logging
	C. AWS Audit Manager - For assessing compliance, not for API logging
	E. Amazon S3 Standard - More expensive for long-term storage compared to Intelligent-Tiering
	Resource URL:
	https://docs.aws.amazon.com/bedrock/latest/userguide/logging-using-cloudtrail.html
	https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html

Q128
A hospital is developing an AI system to assist doctors in diagnosing diseases based on patient records and medical images. To comply with regulations, the sensitive patient data must not leave the country the data is located in.  
Which data governance strategy will ensure compliance and protect patient privacy?
Options:
A. Data residency
B. Data quality
C. Data discoverability
D. Data enrichment

Correct Answer: 
	A. Data residency
	--- Domain 5: Security, Compliance, and Governance for AI Solutions, Task Statement 5.2: "Recognize governance and compliance regulations for AI systems." ---
	Explanation: 
	Answer A is correct because:
	- Data residency ensures that sensitive data remains within specific geographic boundaries
	- It's a key requirement for healthcare compliance and regulations
	- Addresses the requirement that patient data must not leave the country
	- Helps maintain compliance with local healthcare data protection laws
	- Critical for protecting patient privacy in healthcare applications
	The other answers are not suitable because:
	B. Data quality - While important, doesn't address geographic data location requirements
	C. Data discoverability - Focuses on finding and accessing data, not protecting its location
	D. Data enrichment - Relates to improving data value, not compliance or privacy protection
	Resource URL:
	https://docs.aws.amazon.com/prescriptive-guidance/latest/strategy-aws-semicon-workloads/meeting-data-residency-requirements.html
