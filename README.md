MNIST Handwritten Digit Generator
A web application that generates synthetic handwritten digits using a Conditional Generative Adversarial Network (cGAN) trained on the MNIST dataset.

🚀 Features
Interactive Web Interface: Easy-to-use Streamlit app
Custom Trained Model: GAN trained from scratch on MNIST dataset
Real-time Generation: Generate 5 unique digit images with one click
All Digits Supported: Generate any digit from 0-9
Responsive Design: Works on desktop and mobile devices
📁 Project Structure
mnist-digit-generator/
│
├── mnist_gan_training.py    # Training script for the GAN model
├── streamlit_app.py         # Web application using Streamlit
├── requirements.txt         # Python dependencies
├── generator.pth           # Trained generator model (after training)
├── discriminator.pth       # Trained discriminator model (after training)
└── README.md              # This file
🔧 Installation & Setup
1. Clone the Repository
bash
git clone <your-repo-url>
cd mnist-digit-generator
2. Install Dependencies
bash
pip install -r requirements.txt
3. Train the Model (Google Colab Recommended)
Open Google Colab and enable GPU runtime (Runtime → Change runtime type → GPU → T4)
Upload mnist_gan_training.py to Colab
Run the training script:
python
!python mnist_gan_training.py
Download the generated model files: generator.pth and discriminator.pth
Place these files in your project directory
4. Run the Web Application
bash
streamlit run streamlit_app.py
The app will be available at http://localhost:8501

🌐 Deployment Options
Streamlit Cloud (Recommended)
Push your code to GitHub
Go to share.streamlit.io
Connect your GitHub repository
Deploy with streamlit_app.py as the main file
Heroku
Create a Procfile:
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
Deploy to Heroku:
bash
heroku create your-app-name
git push heroku main
🤖 Model Architecture
Generator
Input: 100-dimensional noise vector + digit label (0-9)
Architecture: Fully connected layers with BatchNorm and LeakyReLU
Output: 28x28 grayscale image
Parameters: ~1.4M parameters
Discriminator
Input: 28x28 image + digit label
Architecture: Fully connected layers with Dropout and LeakyReLU
Output: Real/fake classification probability
Parameters: ~2.1M parameters
Training Details
Dataset: MNIST (60,000 training images)
Loss Function: Binary Cross Entropy (adversarial loss)
Optimizer: Adam (lr=0.0002, β1=0.5)
Batch Size: 128
Epochs: 50
Training Time: ~30-60 minutes on T4 GPU
📊 Usage
Select a digit (0-9) from the sidebar dropdown
Click "Generate Images" to create 5 new samples
View the results - each image is a unique generation
Try different digits to explore the model's capabilities
🎯 Performance Requirements
The model generates digits that are:

✅ Recognizable as handwritten digits
✅ Diverse between generations (not identical copies)
✅ Clearly identifiable as the requested digit class
✅ Similar in style to MNIST dataset (28x28 grayscale)
🔍 Technical Specifications
Framework: PyTorch for model, Streamlit for web interface
Model Type: Conditional Generative Adv
