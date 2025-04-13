import React, { useState, useEffect } from 'react';
import { Container, Alert, Select, Box, Button, ContentLayout, Header, SelectProps } from '@cloudscape-design/components';
import Quiz from './components/Quiz';
import Login from './components/Login';
import { parseMarkdownQuiz } from './utils/markdownParser';
import { QuizQuestion } from './types/quiz';
import '@cloudscape-design/global-styles/index.css';

interface QuizFile {
  value: string;
  label: string;
  description?: string;
}

const ALL_DOMAINS_OPTION: QuizFile = {
  value: 'all',
  label: 'All Domains',
  description: 'Show questions from all domains'
};

const DOMAIN_FILES: QuizFile[] = [
  {
    value: '/Domain 1 Questions.md',
    label: 'Domain 1',
    description: 'Design Resilient Architectures'
  },
  { 
    label: 'Domain 2: Fundamentals of Generative AI', 
    value: 'Domain 2 Questions.md',
    description: 'Foundation models, LLMs, and generative AI concepts'
  },
  { 
    label: 'Domain 3: Applications of Foundation Models', 
    value: 'Domain 3 Questions.md',
    description: 'Practical applications and implementations of foundation models'
  },
  { 
    label: 'Domain 4: Guidelines for Responsible AI', 
    value: 'Domain 4 Questions.md',
    description: 'Ethics, bias, transparency, and responsible AI practices'
  },
  { 
    label: 'Domain 5: Security, Compliance, and Governance', 
    value: 'Domain 5 Questions.md',
    description: 'Security best practices and compliance for AI solutions'
  }
];

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<QuizFile>(ALL_DOMAINS_OPTION);
  const [availableFiles, setAvailableFiles] = useState<QuizFile[]>([]);

  const handleLogin = (username: string, password: string) => {
    if (username === 'learner01' && password === '1willpass') {
      setIsAuthenticated(true);
    }
  };

  const handleFileSelect = (option: QuizFile) => {
    setLoading(true);
    setError(null);
  };

  // Load available quiz files
  useEffect(() => {
    if (!isAuthenticated) return;

    setLoading(true);
    console.log('Fetching quiz files...');
    
    fetch('/api/quiz-files')
      .then(async response => {
        const text = await response.text();
        console.log('Raw API response:', text);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch quiz files: ${response.status} ${response.statusText}`);
        }
        
        try {
          return JSON.parse(text);
        } catch (error) {
          if (error instanceof Error) {
            throw new Error(`Invalid JSON response: ${error.message}`);
          }
          throw new Error('Invalid JSON response');
        }
      })
      .then(files => {
        console.log('Received quiz files:', files);
        if (!Array.isArray(files)) {
          throw new Error('Expected array of files but got: ' + typeof files);
        }
        
        setAvailableFiles(files);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading quiz files:', err);
        setError('Failed to load quiz files: ' + err.message);
        setLoading(false);
      });
  }, [isAuthenticated]);

  // Load questions when a file is selected
  useEffect(() => {
    if (!isAuthenticated) return;

    if (selectedFile.value === 'all') {
      // Load all questions from all files
      console.log('Loading all questions...');
      setLoading(true);
      setError(null);

      const loadQuestions = async () => {
        setLoading(true);
        setError(null);
        try {
          const files = [
            'Domain 1 Questions.md',
            'Domain 2 Questions.md',
            'Domain 3 Questions.md',
            'Domain 4 Questions.md',
            'Domain 5 Questions.md'
          ];
          
          const allQuestions: QuizQuestion[] = [];
          console.log('Loading question files...');
          
          for (const file of files) {
            console.log(`Loading ${file}...`);
            const response = await fetch(`/${file}`);
            if (!response.ok) {
              throw new Error(`Failed to load ${file}: ${response.statusText}`);
            }
            const markdown = await response.text();
            console.log(`Parsing ${file}...`);
            const parsedQuestions = parseMarkdownQuiz(markdown);
            console.log(`Found ${parsedQuestions.length} questions in ${file}`);
            allQuestions.push(...parsedQuestions);
          }

          if (allQuestions.length === 0) {
            throw new Error('No valid questions found in any file');
          }

          console.log(`Total questions loaded: ${allQuestions.length}`);
          setQuestions(allQuestions);
        } catch (error) {
          console.error('Error loading questions:', error);
          setError(error instanceof Error ? error.message : 'Failed to load questions');
        } finally {
          setLoading(false);
        }
      };

      loadQuestions();
    } else {
      // Load questions from the selected file
      console.log('Loading questions from:', selectedFile.value);
      setLoading(true);
      setError(null);

      fetch(`/${selectedFile.value}`)
        .then(response => response.text())
        .then(markdown => {
          const questions = parseMarkdownQuiz(markdown);
          console.log('Questions loaded:', questions.length);
          setQuestions(questions);
          setLoading(false);
        })
        .catch(err => {
          console.error('Error loading questions:', err);
          setError('Failed to load questions: ' + err.message);
          setLoading(false);
        });
    }
  }, [selectedFile, availableFiles, isAuthenticated]);

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Container>
      <div className="app-container">
        <header>
          <h1>AWS AI Quiz</h1>
          <Select
            selectedOption={selectedFile}
            onChange={({ detail }) => handleFileSelect(detail.selectedOption as QuizFile)}
            options={[ALL_DOMAINS_OPTION, ...DOMAIN_FILES]}
            selectedAriaLabel="Selected"
            placeholder="Choose a domain"
            triggerVariant="option"
          />
        </header>

        {loading ? (
          <Alert type="info">
            Loading questions...
            <div style={{ marginTop: '1rem', fontSize: '0.9em', color: '#666' }}>
              {selectedFile.label === 'All Domains' ? 'Loading all questions' : `Loading ${selectedFile.label}`}
            </div>
          </Alert>
        ) : error ? (
          <Alert type="error" header="Error loading questions">
            {error}
            <div style={{ marginTop: '1rem' }}>
              Please check the console for more details.
            </div>
          </Alert>
        ) : questions.length > 0 ? (
          <Quiz questions={questions} />
        ) : (
          <Alert type="warning" header="No questions found">
            {selectedFile.label === 'All Domains' 
              ? 'No questions were loaded. Please check the console for more details.'
              : `No questions found for ${selectedFile.label}. Please check the console for more details.`}
          </Alert>
        )}
      </div>
    </Container>
  );
};

export default App; 