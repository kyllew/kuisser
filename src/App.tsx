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
    // Check if username matches pattern learnerXX where XX is 01-30
    const learnerMatch = username.match(/^learner(\d{2})$/);
    if (learnerMatch) {
      const learnerNumber = parseInt(learnerMatch[1]);
      // Check if learner number is between 01-30 and password matches pattern
      if (learnerNumber >= 1 && learnerNumber <= 30 && password === `${learnerNumber}willpass`) {
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