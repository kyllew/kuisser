import { useState, useEffect } from 'react';
import { Container, Alert, Select } from '@cloudscape-design/components';
import Quiz from './components/Quiz';
import { parseMarkdownQuiz } from './utils/markdownParser';
import { QuizFile, Question } from './types/quiz';

const ALL_DOMAINS_OPTION: QuizFile = {
  value: 'all',
  label: 'All Domains',
  description: 'Questions from all domains'
};

const DOMAIN_FILES: QuizFile[] = [
  {
    value: 'Domain 1 Questions.md',
    label: 'Domain 1',
    description: 'Questions about Domain 1'
  },
  {
    value: 'Domain 2 Questions.md',
    label: 'Domain 2',
    description: 'Questions about Domain 2'
  },
  {
    value: 'Domain 3 Questions.md',
    label: 'Domain 3',
    description: 'Questions about Domain 3'
  },
  {
    value: 'Domain 4 Questions.md',
    label: 'Domain 4',
    description: 'Questions about Domain 4'
  }
];

export default function App() {
  const [selectedFile] = useState<QuizFile>(ALL_DOMAINS_OPTION);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = () => {
    // Implementation here if needed
  };

  // Load questions when a file is selected
  useEffect(() => {
    if (!isAuthenticated) return;

    if (selectedFile.value === 'all') {
      // Load all questions from all files
      console.log('Loading all questions...');
      setIsLoading(true);
      setError(null);

      const loadQuestions = async () => {
        setIsLoading(true);
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
          setIsLoading(false);
        }
      };

      loadQuestions();
    } else {
      // Load questions from the selected file
      console.log('Loading questions from:', selectedFile.value);
      setIsLoading(true);
      setError(null);

      fetch(`/${selectedFile.value}`)
        .then(response => response.text())
        .then(markdown => {
          const questions = parseMarkdownQuiz(markdown);
          console.log('Questions loaded:', questions.length);
          setQuestions(questions);
          setIsLoading(false);
        })
        .catch(err => {
          console.error('Error loading questions:', err);
          setError('Failed to load questions: ' + err.message);
          setIsLoading(false);
        });
    }
  }, [selectedFile, isAuthenticated]);

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
            onChange={({ detail }) => handleFileSelect()}
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
} 