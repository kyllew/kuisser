import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readdirSync, readFileSync } from 'fs';
import cors from 'cors';
import { config } from 'dotenv';

config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// API endpoint to get list of quiz files
app.get('/api/quiz-files', (req, res) => {
  const publicDir = join(__dirname, '../public');
  console.log('Reading quiz files from:', publicDir);
  
  try {
    if (!existsSync(publicDir)) {
      console.error('Public directory does not exist:', publicDir);
      return res.status(500).json({ error: 'Public directory not found' });
    }

    const files = readdirSync(publicDir);
    console.log('All files found:', files);
    
    const mdFiles = files.filter(file => file.endsWith('.md'));
    console.log('Markdown files found:', mdFiles);
    
    if (mdFiles.length === 0) {
      console.warn('No markdown files found in directory');
    }
    
    const quizFiles = mdFiles.map(file => ({
      label: file.replace('.md', ''),
      value: file
    }));
    console.log('Sending quiz files:', quizFiles);
    
    res.json(quizFiles);
  } catch (err) {
    console.error('Error reading directory:', err);
    res.status(500).json({ error: 'Failed to read quiz files', details: err.message });
  }
});

// Serve the markdown file
app.get('/:filename.md', (req, res) => {
  const mdPath = join(__dirname, '../public', `${req.params.filename}.md`);
  console.log('Attempting to serve markdown file:', mdPath);
  
  try {
    if (!existsSync(mdPath)) {
      console.error('Markdown file not found:', mdPath);
      return res.status(404).send('Markdown file not found');
    }

    const content = readFileSync(mdPath, 'utf8');
    console.log('Successfully read markdown file, length:', content.length);
    console.log('First 200 characters:', content.substring(0, 200));

    res.type('text/markdown').send(content);
  } catch (err) {
    console.error('Error serving markdown file:', err);
    res.status(500).send('Error reading markdown file: ' + err.message);
  }
});

// Serve static files from the React app dist directory
app.use(express.static(join(__dirname, '../dist')));

// Serve files from public directory
app.use(express.static(join(__dirname, '../public')));

// Routes
app.get('/api/test', (req, res) => {
  res.json({ message: 'Server is running!' });
});

// Handle React routing, return all requests to React app
app.get('*', (req, res) => {
  res.sendFile(join(__dirname, '../dist', 'index.html'));
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port: ${PORT}`);
  console.log('Public directory:', join(__dirname, '../public'));
  
  // List available files on startup
  try {
    const publicDir = join(__dirname, '../public');
    const files = readdirSync(publicDir);
    console.log('Available files on startup:', files);
  } catch (err) {
    console.error('Error listing files on startup:', err);
  }
}); 