import React, { useState } from 'react';
import { Button, Input, FormField, Container, Alert } from '@cloudscape-design/components';
import '@cloudscape-design/global-styles/index.css';

interface LoginProps {
  onLogin: (username: string, password: string) => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = () => {
    setError(null);

    if (username === 'learner01' && password === '1willpass') {
      onLogin(username, password);
    } else {
      setError('Invalid username or password');
    }
  };

  return (
    <Container>
      <div className="login-container" style={{ 
        maxWidth: '400px', 
        margin: '100px auto', 
        padding: '20px',
        border: '1px solid #d1d5db',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Quiz Login</h2>
        
        {error && (
          <Alert type="error" header="Login Failed">
            {error}
          </Alert>
        )}

        <form onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}>
          <FormField label="Username">
            <Input
              value={username}
              onChange={({ detail }) => setUsername(detail.value)}
              placeholder="Enter username"
              autoFocus
            />
          </FormField>

          <FormField label="Password">
            <Input
              value={password}
              onChange={({ detail }) => setPassword(detail.value)}
              type="password"
              placeholder="Enter password"
            />
          </FormField>

          <div style={{ marginTop: '20px', textAlign: 'center' }}>
            <Button variant="primary" onClick={handleSubmit}>
              Login
            </Button>
          </div>
        </form>
      </div>
    </Container>
  );
};

export default Login; 