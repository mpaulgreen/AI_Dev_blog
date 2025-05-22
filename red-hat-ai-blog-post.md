# Enhancing AI Developer Experience with Red Hat: A Practical Guide to InstructLab, RHEL AI, and OpenShift AI

In today's rapidly evolving technological landscape, artificial intelligence has become a cornerstone of innovation for enterprises across industries. However, developing, deploying, and managing AI applications at scale presents significant challenges that many organizations struggle to overcome. Red Hat's comprehensive AI portfolio—including InstructLab, Red Hat Enterprise Linux (RHEL) AI, and OpenShift AI—offers a powerful solution to these challenges, enabling a seamless developer experience while maintaining enterprise-grade reliability and security.

## The Current State of AI Development Challenges

Before diving into Red Hat's solutions, let's examine the pain points that AI developers commonly face:

1. **Limited access to customizable foundation models**: Many organizations struggle to adapt general-purpose large language models (LLMs) to their specific business domains without extensive ML expertise
2. **Lengthy development cycles**: Taking AI projects from experimentation to production often involves complex handoffs between data scientists and operations teams
3. **Infrastructure complexity**: Managing the specialized hardware and software requirements for AI workloads adds significant overhead
4. **Governance and security concerns**: Ensuring models comply with organizational policies and security standards remains challenging
5. **Deployment and scaling limitations**: Many AI projects fail when transitioning from pilot to production due to scalability issues

Red Hat's AI portfolio addresses these challenges head-on, providing a comprehensive platform for developing, training, deploying, and managing AI applications across hybrid environments.

## Red Hat's AI Portfolio: An Overview

Red Hat offers a complete ecosystem for AI development:

- **InstructLab**: A community-driven project that simplifies LLM tuning and customization
- **RHEL AI**: A foundation model platform combining Granite models with InstructLab in a bootable RHEL image
- **OpenShift AI**: A comprehensive platform for managing the complete AI/ML lifecycle at scale

Let's explore how these components work together through a practical use case.

## Use Case: Building a Customer Support AI Assistant with Domain-Specific Knowledge

For our example, let's consider a telecommunications company that wants to create an AI-powered customer support assistant that can access and utilize their proprietary product knowledge base. This assistant needs to:

1. Handle common customer inquiries about telecom products
2. Answer questions specific to the company's services, policies, and technical specifications
3. Integrate with existing customer management systems
4. Scale to support thousands of concurrent users
5. Run across hybrid environments including on-premises data centers and cloud

### Solution Architecture

Our solution will leverage Red Hat's AI portfolio to create a comprehensive, end-to-end workflow:

1. Use InstructLab to customize a foundation model with telecom-specific knowledge
2. Deploy the model using RHEL AI for experimentation and initial testing
3. Scale and productionize the application with OpenShift AI
4. Implement a RAG (Retrieval Augmented Generation) approach to access up-to-date product information

Let's break down the implementation details.

## Step 1: Customizing the Foundation Model with InstructLab

InstructLab provides a user-friendly approach to enhancing LLMs with specific domain knowledge without requiring extensive machine learning expertise.

### Installation and Setup

First, let's install and set up InstructLab with Python virtual environment:

```bash
# Create a new directory for our project
mkdir telecom-ai && cd telecom-ai

# Create a Python virtual environment
python3 -m venv --upgrade-deps venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clear pip cache (recommended)
pip cache remove llama_cpp_python

# Install InstructLab
pip install instructlab

# Initialize InstructLab configuration
ilab config init
```

During the initialization, InstructLab will ask for the path to the taxonomy repository and model locations. You can accept the defaults by pressing Enter.

### Creating the Telecom Knowledge Taxonomy

Next, we'll create a taxonomy of telecommunications knowledge. Create the necessary directories and files:

```bash
# Create the required directory structure
mkdir -p ~/.local/share/instructlab/taxonomy/knowledge/telecom
```

Create a file called `qna.yaml` in this directory with the following content:

```yaml
version: 3
domain: telecom
document_outline: Overview of telecommunications technologies and services
created_by: mrigankapaul
seed_examples:
  - context: |
      Fiber optic internet is a broadband connection that uses fiber optic cables made of thin strands
      of glass or plastic to transmit data using light. These cables can transmit data at speeds up to
      10 Gbps or higher, making them significantly faster than traditional copper connections.
    questions_and_answers:
      - question: What is fiber optic internet?
        answer: |
          Fiber optic internet is a broadband connection that uses fiber optic cables made of thin strands
          of glass or plastic to transmit data using light signals instead of electrical signals.
      - question: What materials are fiber optic cables made from?
        answer: Fiber optic cables are made from thin strands of glass or plastic that can transmit light signals.
      - question: How fast can fiber optic internet transmit data?
        answer: |
          Fiber optic internet can transmit data at speeds up to 10 Gbps or higher, which is significantly
          faster than traditional copper-based connections.
  - context: |
      5G is the fifth generation of cellular network technology, offering significantly faster data
      transmission speeds, lower latency, and greater capacity than previous generations. 5G networks
      operate on higher frequency bands and use smaller cell sites than 4G networks.
    questions_and_answers:
      - question: What is 5G technology?
        answer: 5G is the fifth generation of cellular network technology that offers faster speeds, lower latency, and greater capacity than previous generations.
      - question: How does 5G differ from previous cellular network generations?
        answer: 5G operates on higher frequency bands, uses smaller cell sites, and provides significantly faster data transmission speeds and lower latency than 4G and earlier generations.
      - question: What are the advantages of 5G for mobile users?
        answer: 5G provides mobile users with dramatically faster download and upload speeds, more reliable connections, and the ability to connect more devices simultaneously.
  - context: |
      VoIP (Voice over Internet Protocol) is a technology that allows voice calls to be made over
      internet connections rather than traditional phone lines. It converts analog voice signals into
      digital data packets that are transmitted over IP networks.
    questions_and_answers:
      - question: What is VoIP technology?
        answer: VoIP (Voice over Internet Protocol) is a technology that allows voice calls to be made over internet connections rather than traditional phone lines.
      - question: How does VoIP work?
        answer: VoIP works by converting analog voice signals into digital data packets that are transmitted over IP networks, then converted back to voice at the receiving end.
      - question: What are the benefits of using VoIP over traditional phone services?
        answer: VoIP offers benefits such as lower costs (especially for long-distance calls), greater flexibility, enhanced features like video conferencing, and the ability to make calls from multiple devices.
  - context: |
      Satellite internet provides connectivity through communications satellites rather than land-based
      cables. It's particularly valuable for remote areas where traditional infrastructure is limited or
      absent. Modern satellite internet services can offer download speeds of up to 100 Mbps.
    questions_and_answers:
      - question: What is satellite internet?
        answer: Satellite internet is a connectivity solution that transmits data through communications satellites orbiting Earth rather than through land-based cables.
      - question: Where is satellite internet most valuable?
        answer: Satellite internet is most valuable in remote or rural areas where traditional cable or fiber infrastructure is limited or entirely absent.
      - question: What speeds can modern satellite internet achieve?
        answer: Modern satellite internet services can offer download speeds of up to 100 Mbps, though they typically have higher latency than land-based connections.
  - context: |
      Network latency refers to the time delay between when data is sent and when it's received across
      a network, typically measured in milliseconds. Low latency is crucial for applications requiring
      real-time interaction, such as video calls, online gaming, and industrial control systems.
    questions_and_answers:
      - question: What is network latency?
        answer: Network latency is the time delay between when data is sent and when it's received across a network, typically measured in milliseconds.
      - question: Why is low latency important in telecommunications?
        answer: Low latency is important for applications requiring real-time interaction, such as video calls, online gaming, financial trading, and industrial control systems.
      - question: What factors can increase network latency?
        answer: Network latency can be increased by physical distance between endpoints, network congestion, routing equipment performance, and the use of certain transmission media like satellite connections.
document:
  repo: https://github.com/instructlab/taxonomy.git
  commit: main
  patterns:
    - README.md
```

Be careful about formatting when creating this file:
- Make sure there are no trailing spaces at the end of lines
- Ensure proper indentation
- Add a newline at the end of the file

### Verify and Process the Taxonomy

Now, let's verify our taxonomy is properly formatted:

```bash
# Verify our taxonomy changes
ilab taxonomy diff
```

If all is well, you'll see a message indicating your taxonomy is valid. Next, we'll generate synthetic training data and train the model:

```bash
# Download a model to work with (if not already done during initialization)
ilab model download

# Generate synthetic training data
ilab data generate

# Train the model
ilab model train

# Convert model for macOS compatibility (required for Apple Silicon)
ilab model convert --model-dir ~/.local/share/instructlab/checkpoints/instructlab-granite-7b-lab-mlx-q
```

The training process may take some time depending on your hardware. Once complete, we can test the enhanced model.

InstructLab's approach enables us to efficiently improve the model's understanding of telecommunications concepts with minimal training data and computing resources.

### Testing Your Fine-Tuned Model

After completing the training and conversion, you can test your model using the chat interface:

```bash
ilab model chat --model ./instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
```

This starts an interactive session where you can evaluate how well your model has learned the telecom domain. Here are some effective questions to test your model with:

1. **"What is fiber optic internet and how fast can it be?"**
   - Tests if the model can combine information from multiple related question-answer pairs

2. **"How does 5G technology compare to previous generations?"**
   - Checks the model's understanding of comparative technological information

3. **"Can you explain how VoIP technology works and its advantages?"**
   - Tests if the model can synthesize technical functionality with business benefits

4. **"Why might someone in a rural area consider satellite internet?"**
   - Evaluates the model's understanding of use case scenarios for specific technologies

5. **"What telecommunications technology would be best for real-time gaming?"**
   - Application question requiring the model to consider latency requirements across technologies

6. **"What's the difference between fiber and satellite internet?"**
   - Tests cross-context learning by requiring comparison between two different technology sections

Start with questions directly from your seed examples, then gradually introduce questions requiring synthesis across multiple contexts. This progression will help you evaluate both knowledge retention and the model's ability to generalize from its training.

## Step 2: Model Deployment with RHEL AI [Verification in Progress for this step and below]

With our enhanced model ready, we'll deploy it using RHEL AI for initial testing and validation. RHEL AI provides a bootable image with everything needed to run and serve our model.

### Setting Up RHEL AI on Mac with M3 Chip

For Mac users with Apple Silicon chips, here's a detailed guide to getting RHEL AI running:

1. **Download RHEL AI**:
   - Go to the [Red Hat Developer portal](https://developers.redhat.com/products/rhel-ai/overview) and download the RHEL AI bootable image for ARM architecture
   - You'll need a Red Hat account (free for developers)

2. **Choose a Virtualization Solution**:
   Several options work well with Apple Silicon:
   - VMware Fusion (paid solution with good performance)
   - Parallels Desktop (paid solution with excellent Apple Silicon integration)
   - UTM (free and open-source solution based on QEMU)

3. **Create a Virtual Machine**:
   ```
   # Using VMware Fusion as an example
   1. Open VMware Fusion
   2. Click "Create a New Virtual Machine"
   3. Choose "Install from disc or image"
   4. Select the RHEL AI ISO file
   5. Select "Linux" -> "Red Hat Enterprise Linux 9 (ARM)"
   6. Configure resources (recommended: 8+ CPU cores, 16+ GB RAM, 128+ GB storage)
   7. Complete VM creation
   ```

4. **Install RHEL AI**:
   - Follow the RHEL installation prompts
   - During installation, choose the AI/ML package collection
   - Complete the setup with appropriate networking configuration

### Transferring Your Enhanced Model

After training a model with InstructLab, you need to transfer it to your RHEL AI environment:

1. **Export Your Model**:
   ```bash
   # On your development machine where you trained the model
   # Identify the model path
   ls ~/.local/share/instructlab/checkpoints
   
   # Archive the model for transfer
   tar -czvf telecom-model.tar.gz ~/.local/share/instructlab/checkpoints/YOUR_MODEL_FOLDER
   ```

2. **Transfer to RHEL AI VM**:
   ```bash
   # Using scp (secure copy)
   scp telecom-model.tar.gz username@rhel-vm-ip:/home/username/
   
   # Or via shared folder if configured in your virtualization software
   ```

3. **Extract on RHEL AI**:
   ```bash
   # SSH into your RHEL AI VM
   ssh username@rhel-vm-ip
   
   # Extract the model
   mkdir -p ~/models
   tar -xzvf telecom-model.tar.gz -C ~/models
   ```

### Deploying the Model with vLLM

RHEL AI includes vLLM (a high-performance inference engine) for efficient model serving:

1. **Set Up the Model Server**:
   ```bash
   # Create a deployment directory
   mkdir -p ~/telecom-assistant
   cd ~/telecom-assistant
   
   # Start the vLLM server with your model
   vllm serve \
     --model ~/models/YOUR_MODEL_PATH \
     --tensor-parallel-size 1 \  # Adjust based on available GPUs
     --host 0.0.0.0 \
     --port 8000
   ```

2. **Create an API Interface**:
   Create a simple API wrapper using Flask to make interacting with the model easier:

   ```python
   from flask import Flask, request, jsonify
   import requests

   app = Flask(__name__)

   MODEL_ENDPOINT = "http://localhost:8000/v1/completions"

   @app.route('/api/support', methods=['POST'])
   def get_support_response():
       query = request.json.get('query', '')
       
       payload = {
           "prompt": f"Customer: {query}\nSupport Assistant:",
           "max_tokens": 500,
           "temperature": 0.7
       }
       
       response = requests.post(MODEL_ENDPOINT, json=payload)
       model_response = response.json()
       
       return jsonify({
           "response": model_response['choices'][0]['text'].strip()
       })

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
   ```

3. **Run the API Wrapper**:
   ```bash
   # Install Flask if needed
   pip install flask requests
   
   # Run the API
   python api.py
   ```

4. **Test Your Deployment**:
   ```bash
   # From another terminal
   curl -X POST http://localhost:5000/api/support \
     -H "Content-Type: application/json" \
     -d '{"query": "Tell me about fiber optic internet"}'
   ```

### Performance Considerations for M3 Macs

When running AI models on Apple Silicon:

1. **Hardware Acceleration**: RHEL AI can leverage the Neural Engine and GPU in Apple Silicon, though performance characteristics differ from NVIDIA GPUs.

2. **Memory Management**: M3 Macs with unified memory architecture efficiently handle models, but monitoring memory usage is important.

3. **Quantization**: Consider using quantized versions of your models for better performance. RHEL AI supports various quantization techniques to reduce model size and improve inference speed.

This setup provides a development environment for testing your model's responses and making iterative improvements before scaling with OpenShift AI.

## Step 3: Scaling with OpenShift AI

Once we've validated our model's performance, we'll leverage OpenShift AI to productionize and scale our solution:

1. Deploy OpenShift cluster with GPU support
2. Install OpenShift AI operator
3. Create a model server deployment:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: telecom-support-assistant
  namespace: ai-models
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      runtime: triton
      storageUri: pvc://model-storage/telecom-assistant
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          memory: "4Gi"
          cpu: "1"
```

4. Deploy our front-end application that will interact with the model:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: support-assistant-app
  namespace: ai-apps
spec:
  replicas: 3
  selector:
    matchLabels:
      app: support-assistant
  template:
    metadata:
      labels:
        app: support-assistant
    spec:
      containers:
      - name: support-assistant
        image: registry.example.com/support-assistant:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_ENDPOINT
          value: "http://telecom-support-assistant.ai-models.svc.cluster.local"
```

## Step 4: Implementing RAG for Up-to-Date Information

To ensure our assistant has access to the latest product information, we'll implement a Retrieval Augmented Generation (RAG) system using OpenShift AI's capabilities:

1. Deploy a vector database (e.g., Redis) to store embeddings of our knowledge base documents:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-vector-db
  namespace: ai-infra
spec:
  serviceName: "redis"
  replicas: 1
  selector:
    matchLabels:
      app: redis-vector
  template:
    metadata:
      labels:
        app: redis-vector
    spec:
      containers:
      - name: redis
        image: redislabs/redisearch:latest
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 20Gi
```

2. Create a data pipeline to process and embed documents into the vector database:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Redis
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = DirectoryLoader('./knowledge_base', glob="**/*.md")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and store embeddings in Redis
redis_url = "redis://redis-vector-db.ai-infra:6379"
vectorstore = Redis.from_documents(
    texts, 
    embeddings, 
    redis_url=redis_url,
    index_name="telecom_knowledge"
)
```

3. Enhance our API to use RAG for responding to queries:

```python
from flask import Flask, request, jsonify
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Redis

app = Flask(__name__)

MODEL_ENDPOINT = "http://telecom-support-assistant.ai-models.svc.cluster.local/v1/completions"
REDIS_URL = "redis://redis-vector-db.ai-infra:6379"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Redis(redis_url=REDIS_URL, index_name="telecom_knowledge", embedding=embeddings)

@app.route('/api/support', methods=['POST'])
def get_support_response():
    query = request.json.get('query', '')
    
    # Retrieve relevant context from vector database
    relevant_docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt with context
    prompt = f"""
    Context information:
    {context}
    
    Customer: {query}
    Support Assistant:
    """
    
    # Query the model
    payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    response = requests.post(MODEL_ENDPOINT, json=payload)
    model_response = response.json()
    
    return jsonify({
        "response": model_response['choices'][0]['text'].strip()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Benefits of the Red Hat Approach

Our implementation leverages the strengths of each component in Red Hat's AI portfolio:

1. **InstructLab** enables domain experts to enhance the model's knowledge without deep ML expertise
2. **RHEL AI** provides a streamlined development environment for initial model testing
3. **OpenShift AI** delivers enterprise-grade scalability, security, and management capabilities
4. The overall architecture supports a hybrid deployment model that can run anywhere

Key advantages include:

- **Reduced time-to-market**: The streamlined workflow significantly accelerates development cycles
- **Domain customization**: The ability to enhance models with proprietary knowledge creates a competitive advantage
- **Operational consistency**: The same platform can be used from development to production
- **Enterprise readiness**: Security, monitoring, and governance capabilities are built in
- **Cost efficiency**: Open source foundation reduces licensing costs and vendor lock-in

## Common Issues and Troubleshooting

When working with InstructLab, you might encounter some common issues:

### Taxonomy Validation Errors

InstructLab has strict requirements for taxonomy files:
- **No trailing spaces**: Make sure there are no spaces at the end of lines in your YAML files
- **Taxonomy version**: Use version 3 for knowledge taxonomies
- **Required fields**: Ensure all required fields (domain, document, questions_and_answers) are present
- **Minimum examples**: Knowledge taxonomies require at least 5 seed examples
- **Repository references**: The document section must reference a valid GitHub repository

To check for these issues, run:
```bash
ilab taxonomy diff
```

If you encounter validation errors, carefully review the error messages and fix each issue.

### Environment Setup Issues

If you're missing tools like `yq`:
```bash
# For macOS
brew install yq

# For Ubuntu/Debian
sudo apt-get install yq

# For Fedora/RHEL
sudo dnf install yq
```

### Model Training Performance

For better performance during model training:
- Use GPU acceleration when available
- Start with smaller datasets for initial testing
- Consider using OpenShift AI for distributed training at scale

### macOS Model Compatibility

If you encounter errors about vLLM not supporting your platform, remember to convert your model to GGUF format:
```bash
ilab model convert --model-dir ~/.local/share/instructlab/checkpoints/YOUR_MODEL
```

## Conclusion

Red Hat's AI portfolio represents a comprehensive approach to addressing the challenges of developing and deploying AI applications at scale. Through the combined capabilities of InstructLab, RHEL AI, and OpenShift AI, organizations can streamline the entire AI lifecycle—from model customization to production deployment and ongoing management.

The telecommunications customer support example demonstrates how these tools can be integrated to create a practical, scalable solution that delivers real business value. By leveraging Red Hat's open source approach, organizations can accelerate their AI initiatives while maintaining the security, reliability, and flexibility required for enterprise deployments.

As AI continues to transform industries, having a robust, scalable, and flexible foundation for development and deployment will be crucial for success. Red Hat's AI portfolio provides exactly that foundation, enabling organizations to innovate faster and more effectively in an increasingly AI-driven world.

## Next Steps

To get started with Red Hat's AI portfolio:

1. **Explore InstructLab**: Visit [instructlab.ai](https://instructlab.ai) and try out the community version
2. **Try RHEL AI**: Download the developer preview from Red Hat's website
3. **Deploy OpenShift AI**: Contact Red Hat for a trial of OpenShift AI
4. **Join the community**: Contribute to the InstructLab taxonomy repository to improve the models

By leveraging these powerful tools together, you can create sophisticated AI applications that meet the needs of your organization while maintaining enterprise-grade reliability and security.