import { useState, useEffect } from 'react';
import { Container, Alert, Select, FormField, Input, Button } from '@cloudscape-design/components';
import Quiz from './components/Quiz';
import { parseMarkdownQuiz } from './utils/markdownParser';
import { QuizFile, Question } from './types/quiz';
import './App.css';

const ALL_DOMAINS_OPTION: QuizFile = {
  value: 'all',
  label: 'All Domains',
  description: 'Questions from all domains'
};

const DOMAIN_FILES: QuizFile[] = [
  {
    value: 'Domain 1 Questions.md',
    label: 'Domain 1: Fundamentals of AI and ML',
    description: 'Core concepts of artificial intelligence and machine learning'
  },
  {
    value: 'Domain 2 Questions.md',
    label: 'Domain 2: Fundamentals of Generative AI',
    description: 'Foundation models, LLMs, and generative AI concepts'
  },
  {
    value: 'Domain 3 Questions.md',
    label: 'Domain 3: Applications of Foundation Models',
    description: 'Practical applications and implementations of foundation models'
  },
  {
    value: 'Domain 4 Questions.md',
    label: 'Domain 4: Guidelines for Responsible AI',
    description: 'Ethics, bias, transparency, and responsible AI practices'
  },
  {
    value: 'Domain 5 Questions.md',
    label: 'Domain 5: Security, Compliance, and Governance for AI Solutions',
    description: 'Security best practices and compliance for AI solutions'
  }
];

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    const savedAuth = localStorage.getItem('auth-state');
    return savedAuth ? JSON.parse(savedAuth).isAuthenticated : false;
  });
  const [username, setUsername] = useState(() => {
    const savedAuth = localStorage.getItem('auth-state');
    return savedAuth ? JSON.parse(savedAuth).username : '';
  });
  const [password, setPassword] = useState('');
  const [selectedFile, setSelectedFile] = useState<QuizFile>(ALL_DOMAINS_OPTION);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(() => {
    const savedAuth = localStorage.getItem('auth-state');
    return savedAuth ? JSON.parse(savedAuth).sessionId : '';
  });

  const handleLogin = (username: string, password: string) => {
    // Check if username matches pattern learnerXX where XX is 01-50
    const learnerMatch = username.match(/^learner(\d{2})$/);
    if (learnerMatch) {
      const learnerNumber = parseInt(learnerMatch[1]);
      // Check if learner number is between 01-50 and password matches pattern
      if (learnerNumber >= 1 && learnerNumber <= 50 && password === `${learnerNumber}willpass`) {
        // Use a consistent sessionId based on username only
        const newSessionId = `learner-${learnerNumber}`;
        setSessionId(newSessionId);
        setIsAuthenticated(true);
        setUsername(username);
        // Save auth state to localStorage
        localStorage.setItem('auth-state', JSON.stringify({
          isAuthenticated: true,
          username,
          sessionId: newSessionId
        }));
        return;
      }
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername('');
    setPassword('');
    // Don't clear sessionId or quiz state
    localStorage.setItem('auth-state', JSON.stringify({
      isAuthenticated: false,
      username: '',
      sessionId: sessionId // Keep the sessionId
    }));
  };

  // Load questions when a file is selected
  useEffect(() => {
    if (!isAuthenticated) return;

    const loadQuestions = async () => {
      setError(null);
      try {
        if (selectedFile.value === 'all') {
          // Load all questions from all files
          console.log('Loading all questions...');
          const allQuestions: Question[] = [];
          let totalQuestions = 0;
          const loadedQuestionNumbers = new Set<number>();

          // Load questions from each file sequentially
          for (const file of DOMAIN_FILES) {
            try {
              console.log(`Loading questions from ${file.value}...`);
              const response = await fetch(`/${file.value}`);
              if (!response.ok) {
                throw new Error(`Failed to load ${file.value}: ${response.statusText}`);
              }
              const text = await response.text();
              const questions = parseMarkdownQuiz(text);
              console.log(`Loaded ${questions.length} questions from ${file.value}`);
              
              // Track which question numbers were loaded
              const questionNumbers = text.match(/Q(\d+)/g)?.map(match => parseInt(match.replace('Q', ''))) || [];
              questionNumbers.forEach(qNum => loadedQuestionNumbers.add(qNum));
              
              totalQuestions += questions.length;
              allQuestions.push(...questions);
            } catch (error) {
              console.error(`Error loading ${file.value}:`, error);
              setError(`Failed to load ${file.value}: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
          }

          // Find missing question numbers
          const missingQuestions: number[] = [];
          for (let i = 1; i <= 155; i++) {
            if (!loadedQuestionNumbers.has(i)) {
              missingQuestions.push(i);
            }
          }

          console.log('=== Question Loading Summary ===');
          console.log(`Total questions successfully loaded: ${totalQuestions}`);
          console.log(`Missing questions count: ${missingQuestions.length}`);
          console.log('Missing question numbers:', missingQuestions.map(q => `Q${q}`).join(', '));
          
          setQuestions(allQuestions);
        } else {
          // Load questions from the selected file
          console.log('Loading questions from:', selectedFile.value);
          const response = await fetch(`/${selectedFile.value}`);
          if (!response.ok) {
            throw new Error(`Failed to load ${selectedFile.value}: ${response.statusText}`);
          }
          const text = await response.text();
          const questions = parseMarkdownQuiz(text);
          
          // Track loaded question numbers for single file
          const loadedQuestionNumbers = new Set<number>();
          const questionNumbers = text.match(/Q(\d+)/g)?.map(match => parseInt(match.replace('Q', ''))) || [];
          questionNumbers.forEach(qNum => loadedQuestionNumbers.add(qNum));

          // Find missing question numbers
          const missingQuestions: number[] = [];
          for (let i = 1; i <= 155; i++) {
            if (!loadedQuestionNumbers.has(i)) {
              missingQuestions.push(i);
            }
          }

          console.log('=== Question Loading Summary ===');
          console.log(`Questions loaded from ${selectedFile.value}: ${questions.length}`);
          console.log(`Missing questions count: ${missingQuestions.length}`);
          console.log('Missing question numbers:', missingQuestions.map(q => `Q${q}`).join(', '));
          
          setQuestions(questions);
        }
      } catch (error) {
        console.error('Error loading questions:', error);
        setError(error instanceof Error ? error.message : 'Failed to load questions');
      }
    };

    loadQuestions();
  }, [selectedFile, isAuthenticated]);

  const handleFileSelect = (event: any) => {
    setSelectedFile(event.detail.selectedOption);
  };

  if (!isAuthenticated) {
    return (
      <Container>
        <h1>AI Quiz Demo</h1>
        <div className="login-container">
          <form onSubmit={(e) => {
            e.preventDefault();
            handleLogin(username, password);
          }}>
            <FormField label="Username">
              <Input
                name="username"
                placeholder="Enter your username"
                value={username}
                onChange={({ detail }) => setUsername(detail.value)}
              />
            </FormField>
            <FormField label="Password">
              <Input
                name="password"
                placeholder="Enter your password"
                value={password}
                onChange={({ detail }) => setPassword(detail.value)}
                type="password"
              />
            </FormField>
            <div style={{ marginTop: '20px' }}>
              <Button variant="primary">Login</Button>
            </div>
          </form>
        </div>
      </Container>
    );
  }

  return (
    <Container>
      <h1>AI Quiz Demo</h1>
      
      {isAuthenticated && (
        <div className="user-info">
          <span>Logged in as: {username}</span>
          <Button 
            variant="normal" 
            onClick={handleLogout}
            iconName="external"
          >
            Logout
          </Button>
        </div>
      )}

      <div className="file-selector">
        <Select
          selectedOption={selectedFile}
          onChange={handleFileSelect}
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
        <Quiz 
          questions={questions} 
          sessionId={sessionId}
        />
      )}
    </Container>
  );
} 