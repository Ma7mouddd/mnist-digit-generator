import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Generator Network (same as training script)
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Generator layers
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Combine noise and label embeddings
        label_embed = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embed], dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    device = torch.device('cpu')  # Use CPU for web deployment
    
    # Initialize model
    latent_dim = 100
    num_classes = 10
    generator = Generator(latent_dim, num_classes)
    
    try:
        # Try to load the trained model
        generator.load_state_dict(torch.load('generator.pth', map_location=device))
        generator.eval()
        st.success("✅ Trained model loaded successfully!")
        return generator, True
    except FileNotFoundError:
        st.warning("⚠️ Trained model not found. Using randomly initialized model for demonstration.")
        generator.eval()
        return generator, False

def generate_digit_images(generator, digit, num_images=5):
    """Generate images for a specific digit"""
    latent_dim = 100
    device = torch.device('cpu')
    
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, latent_dim)
        
        # Create labels for the specified digit
        labels = torch.full((num_images,), digit, dtype=torch.long)
        
        # Generate images
        fake_imgs = generator(noise, labels)
        
        # Convert to numpy and denormalize
        fake_imgs = fake_imgs.cpu().numpy()
        fake_imgs = (fake_imgs + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        fake_imgs = np.clip(fake_imgs, 0, 1)
        
        return fake_imgs

def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    # Convert to 0-255 range and uint8
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

def main():
    st.title("🔢 Handwritten Digit Generator")
    st.markdown("Generate synthetic handwritten digits using a trained GAN model")
    
    # Load model
    generator, model_loaded = load_model()
    
    # Create sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate:",
        options=list(range(10)),
        index=2
    )
    
    # Generation button
    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner(f"Generating images of digit {selected_digit}..."):
            # Generate images
            generated_images = generate_digit_images(generator, selected_digit, 5)
            
            # Store in session state
            st.session_state.generated_images = generated_images
            st.session_state.current_digit = selected_digit
    
    # Display results
    if hasattr(st.session_state, 'generated_images'):
        st.header(f"Generated Images of Digit {st.session_state.current_digit}")
        
        # Create columns for image display
        cols = st.columns(5)
        
        for i, col in enumerate(cols):
            with col:
                # Convert numpy array to PIL Image
                img = numpy_to_pil(st.session_state.generated_images[i, 0])
                
                # Resize for better display
                img_resized = img.resize((112, 112), Image.NEAREST)
                
                # Display image
                st.image(img_resized, caption=f"Sample {i+1}", use_column_width=True)
        
        # Display model info
        if model_loaded:
            st.success("✅ Images generated using trained GAN model")
        else:
            st.warning("⚠️ Images generated using untrained model (random weights)")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **How it works:**
        - Uses a Conditional GAN (cGAN) trained on MNIST dataset
        - Generator creates 28x28 grayscale images
        - Each generation produces 5 unique samples
        - Model trained from scratch using PyTorch
        """)
    
    with col2:
        st.markdown("""
        **Model Architecture:**
        - Generator: Fully connected layers with BatchNorm
        - Discriminator: Fully connected with Dropout
        - Label conditioning for digit-specific generation
        - Trained with adversarial loss
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Training Configuration:**
        - Dataset: MNIST (60,000 training images)
        - Latent dimension: 100
        - Batch size: 128
        - Learning rate: 0.0002
        - Optimizer: Adam with β1=0.5
        - Training epochs: 50
        - Hardware: Google Colab T4 GPU
        
        **Model Components:**
        - **Generator**: Takes noise + digit label → generates 28x28 image
        - **Discriminator**: Takes image + digit label → real/fake classification
        - **Loss**: Binary Cross Entropy (adversarial loss)
        """)
    
    # Instructions
    with st.expander("📝 Instructions"):
        st.markdown("""
        1. **Select a digit** (0-9) from the sidebar dropdown
        2. **Click "Generate Images"** to create 5 new samples
        3. **View the results** - each image is a unique generation
        4. **Try different digits** to see the model's versatility
        
        **Note:** If the trained model isn't available, the app will use random weights for demonstration purposes.
        """)

if __name__ == "__main__":
    main()