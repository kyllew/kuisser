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
  const [error, setError] = useState<string | null>(null);

  // Load questions when a file is selected
  useEffect(() => {
    if (selectedFile.value === 'all') {
      // Load all questions from all files
      console.log('Loading all questions...');
      setError(null);

      const loadQuestions = async () => {
        setError(null);
        try {
          const allQuestions: Question[] = [];
          for (const file of DOMAIN_FILES) {
            const response = await fetch(`/${file.value}`);
            const text = await response.text();
            const questions = parseMarkdownQuiz(text);
            allQuestions.push(...questions);
          }
          console.log('All questions loaded:', allQuestions.length);
          setQuestions(allQuestions);
        } catch (error) {
          console.error('Error loading questions:', error);
          setError(error instanceof Error ? error.message : 'Failed to load questions');
        }
      };

      loadQuestions();
    } else {
      // Load questions from the selected file
      console.log('Loading questions from:', selectedFile.value);
      setError(null);

      fetch(`/${selectedFile.value}`)
        .then(response => response.text())
        .then(text => {
          const questions = parseMarkdownQuiz(text);
          console.log('Questions loaded:', questions.length);
          setQuestions(questions);
        })
        .catch(err => {
          console.error('Error loading questions:', err);
          setError('Failed to load questions: ' + err.message);
        });
    }
  }, [selectedFile]);

  const handleFileSelect = () => {
    // Implementation here if needed
  };

  return (
    <Container>
      <h1>AWS Certified Machine Learning - Specialty Practice Questions</h1>
      
      <div className="file-selector">
        <Select
          selectedOption={selectedFile}
          onChange={() => handleFileSelect()}
          options={[ALL_DOMAINS_OPTION, ...DOMAIN_FILES]}
          selectedAriaLabel="Selected"
        />
      </div>

      {error && (
        <Alert
          type="error"
          header="Error loading questions"
        >
          {error}
        </Alert>
      )}

      {questions.length > 0 && (
        <Quiz questions={questions} />
      )}
    </Container>
  );
} 