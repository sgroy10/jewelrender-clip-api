const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { pipeline } = require('@xenova/transformers');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Configure Multer to handle file uploads in memory
const upload = multer({ storage: multer.memoryStorage() });

let imageVectorizer = null;

// Load CLIP model once at startup
async function loadCLIP() {
  if (!imageVectorizer) {
    console.log('⏳ Loading CLIP model (Xenova/clip-vit-base-patch32)...');
    imageVectorizer = await pipeline('feature-extraction', 'Xenova/clip-vit-base-patch32');
    console.log('✅ CLIP model ready');
  }
}

// Load model at startup
loadCLIP();

// Utility to convert buffer to base64 data URI
function bufferToDataURL(buffer, mimetype) {
  const base64 = buffer.toString('base64');
  return `data:${mimetype};base64,${base64}`;
}

// POST /vectorize-image
app.post('/vectorize-image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const dataUrl = bufferToDataURL(req.file.buffer, req.file.mimetype);
    const output = await imageVectorizer(dataUrl, { pooling: 'mean', normalize: true });

    res.json({
      success: true,
      vector: Array.from(output.data),
      dimensions: output.data.length
    });
  } catch (err) {
    console.error('❌ Error during vectorization:', err);
    res.status(500).json({ error: 'Failed to vectorize image' });
  }
});

app.get('/', (req, res) => {
  res.send('🧠 CLIP API is running');
});

app.listen(port, () => {
  console.log(`🚀 Server running on http://localhost:${port}`);
});
