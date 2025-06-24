import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Load CLIP model
let clip;
(async () => {
  console.log('🔁 Loading CLIP model...');
  clip = await pipeline('feature-extraction', 'Xenova/clip-vit-base-patch32');
  console.log('✅ CLIP model loaded');
})();

// Root route
app.get('/', (req, res) => {
  res.send('JewelRender CLIP API is running ✅');
});

// Generate embedding from text
app.post('/embed', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text must be a string' });
    }

    const output = await clip(text, { pooling: 'mean', normalize: true });
    res.json({ embedding: Array.from(output.data) });
  } catch (error) {
    console.error('Embedding error:', error);
    res.status(500).json({ error: 'Failed to generate embedding' });
  }
});

// Start server
app.listen(port, () => {
  console.log(`🚀 Server is running at http://localhost:${port}`);
});
