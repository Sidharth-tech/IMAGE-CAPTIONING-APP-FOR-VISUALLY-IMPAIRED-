from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os
import io
from PIL import Image
import base64

# Constants from your original app
MAX_LENGTH = 40
EMBEDDING_DIM = 512
UNITS = 512

# Model components adapted from your original app
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training):
        x = self.norm1(x)
        x = self.dense(x)
        x = self.norm2(x + self.attn(x, x, x, training=training))
        return x

class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, input_ids):
        positions = tf.range(tf.shape(input_ids)[-1])[tf.newaxis, :]
        return self.token_embeddings(input_ids) + self.position_embeddings(positions)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, units, num_heads, vocab_size):
        super().__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids, encoder_output, training=False, mask=None):
        embeddings = self.embedding(input_ids)

        # === Mask handling ===
        causal_mask = self.get_causal_attention_mask(embeddings)
        causal_mask = tf.cast(causal_mask, tf.int32)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            padding_mask = None
            combined_mask = causal_mask

        # === Decoder layers ===
        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )
        out_1 = self.layernorm_1(embeddings + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)

        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        seq_len = tf.shape(inputs)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, axis=0)


class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def call(self, img, input_seq, training=False, mask=None):
        img_embed = self.cnn_model(img)
        enc_output = self.encoder(img_embed, training=training)
        return self.decoder(input_seq, enc_output, training=training, mask=mask)


# Initialize Flask app
app = Flask(__name__)

# Global variables to store model components
model = None
tokenizer = None
word2idx = None
idx2word = None

def load_model():
    """Load the TensorFlow model and associated components"""
    global model, tokenizer, word2idx, idx2word
    
    print("Loading TensorFlow model...")
    
    # Paths to model files
    model_dir = os.path.join(os.getcwd(), 'model')
    vocab_path = os.path.join(model_dir, 'vocab.file')
    model_path = os.path.join(model_dir, 'modelimage.h5')
    
    # Check if model files exist
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model files not found. Place vocab.file and modelimage.h5 in the model directory.")
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create tokenizer
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        output_sequence_length=MAX_LENGTH,
        standardize=None
    )
    tokenizer.set_vocabulary(vocab)
    
    # Create word lookup layers
    word2idx = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token="")
    idx2word = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token="", invert=True)
    
    # Build model components
    cnn_model = _build_cnn_encoder()
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, num_heads=1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8, len(vocab))
    model = ImageCaptioningModel(cnn_model, encoder, decoder)
    
    # Build model with dummy call
    dummy_img = tf.zeros((1, 299, 299, 3))
    dummy_seq = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    _ = model(dummy_img, dummy_seq)
    
    # Load weights
    model.load_weights(model_path)
    
    print("Model loaded successfully")

def _build_cnn_encoder():
    """Create CNN encoder model"""
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    output = tf.keras.layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    return tf.keras.models.Model(base_model.input, output)

def preprocess_image(img_array):
    """Preprocess image for the model"""
    img = cv2.resize(img_array, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def generate_caption(img_array):
    """Generate caption for an image"""
    global model, tokenizer, word2idx, idx2word
    
    try:
        img = preprocess_image(img_array)
        img = tf.expand_dims(img, axis=0)
        img_embed = model.cnn_model(img)
        enc_out = model.encoder(img_embed, training=False)

        y_input = '[start]'
        for i in range(MAX_LENGTH - 1):
            tokenized = tokenizer([y_input])[:, :-1]
            mask = tf.cast(tokenized != 0, tf.int32)
            predictions = model.decoder(tokenized, enc_out, training=False, mask=mask)
            next_id = tf.argmax(predictions[0, i]).numpy()
            next_word = idx2word(next_id).numpy().decode('utf-8')
            if next_word == '[end]':
                break
            y_input += ' ' + next_word
            
        return y_input.replace('[start] ', '')
    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/caption', methods=['POST'])
def caption_image():
    """Endpoint to receive image and return caption"""
    if 'image' not in request.files:
        # Check if image is sent as base64 in JSON
        if request.json and 'image' in request.json:
            try:
                # Decode base64 image
                image_data = base64.b64decode(request.json['image'])
                img = Image.open(io.BytesIO(image_data))
                img_array = np.array(img)
                
                # Convert to BGR if image is RGB (for OpenCV compatibility)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": f"Error decoding base64 image: {str(e)}"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400
    else:
        try:
            # Get image file from request
            image_file = request.files['image']
            img = Image.open(image_file)
            img_array = np.array(img)
            
            # Convert to BGR if image is RGB (for OpenCV compatibility)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({"error": f"Error processing image file: {str(e)}"}), 400
    
    # Generate caption
    caption = generate_caption(img_array)
    
    # Return caption
    return jsonify({
        "caption": caption
    })

if __name__ == '__main__':
    # Load model when server starts
    load_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000)