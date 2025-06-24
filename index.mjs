import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

let imageVectorizer = null;
let textVectorizer = null;

async function loadCLIP() {
  if (!imageVectorizer || !textVectorizer) {
    console.log('Loading CLIP model...');
    imageVectorizer = await pipeline('feature-extraction', 'Xenova/clip-vit-base-patch32');
    textVectorizer = await pipeline('feature-extraction', 'Xenova/clip-vit-base-patch32');
    console.log('CLIP model loaded.');
  }
}

function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (normA * normB);
}

app.post('/text-embedding', async (req, res) => {
  try {
    const { text } = req.body;
    await loadCLIP();
    const output = await textVectorizer(text, { pooling: 'mean', normalize: true });
    res.json({ embedding: Array.from(output.data) });
  } catch (err) {
    console.error('Text embedding error:', err);
    res.status(500).json({ error: 'Text embedding failed.' });
  }
});

app.post('/image-embedding', async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    await loadCLIP();
    const output = await imageVectorizer(imageBase64, { pooling: 'mean', normalize: true });
    res.json({ embedding: Array.from(output.data) });
  } catch (err) {
    console.error('Image embedding error:', err);
    res.status(500).json({ error: 'Image embedding failed.' });
  }
});

app.listen(port, () => {
  console.log(`✅ CLIP API listening on http://localhost:${port}`);
});
