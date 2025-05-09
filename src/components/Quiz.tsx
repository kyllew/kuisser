import { useState, useEffect } from 'react';
import { Button } from '@cloudscape-design/components';
import './Quiz.css';
import { Question } from '../types/quiz';

interface QuizProps {
  questions: Question[];
  sessionId: string;
}

export default function Quiz({ questions, sessionId }: QuizProps) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<{ [key: number]: string[] }>({});
  const [submitted, setSubmitted] = useState<{ [key: number]: boolean }>({});
  const [score, setScore] = useState(0);
  const [answeredQuestions, setAnsweredQuestions] = useState<Map<number, { selectedAnswers: string[]; isCorrect: boolean }>>(new Map());
  const [showDisclaimer, setShowDisclaimer] = useState<{ [key: number]: boolean }>({});

  // Load saved state for this session
  useEffect(() => {
    console.log('Loading state for session:', sessionId);
    const savedState = localStorage.getItem(`quiz-state-${sessionId}`);
    console.log('Raw saved state:', savedState);
    
    if (savedState) {
      try {
        const parsedState = JSON.parse(savedState);
        console.log('Parsed state:', JSON.stringify(parsedState, null, 2));
        
        const { 
          currentIndex, 
          answers, 
          submittedAnswers, 
          currentScore, 
          answeredQs,
          disclaimer
        } = parsedState;
        
        // Convert answeredQs back to Map
        const answeredQsMap = new Map<number, { selectedAnswers: string[]; isCorrect: boolean }>();
        if (answeredQs) {
          Object.entries(answeredQs).forEach(([key, value]) => {
            answeredQsMap.set(Number(key), value as { selectedAnswers: string[]; isCorrect: boolean });
          });
        }
        
        console.log('Restoring state:', JSON.stringify({
          currentIndex,
          answers,
          submittedAnswers,
          currentScore,
          answeredQs: Object.fromEntries(answeredQsMap),
          disclaimer
        }, null, 2));
        
        setCurrentQuestionIndex(currentIndex);
        setSelectedAnswers(answers || {});
        setSubmitted(submittedAnswers || {});
        setScore(currentScore || 0);
        setAnsweredQuestions(answeredQsMap);
        setShowDisclaimer(disclaimer || {});
      } catch (error) {
        console.error('Error restoring state:', error);
      }
    } else {
      console.log('No saved state found for session:', sessionId);
    }
  }, [sessionId]);

  // Save state whenever it changes
  useEffect(() => {
    console.log('Saving state for session:', sessionId);
    const stateToSave = {
      currentIndex: currentQuestionIndex,
      answers: selectedAnswers,
      submittedAnswers: submitted,
      currentScore: score,
      answeredQs: Object.fromEntries(answeredQuestions),
      disclaimer: showDisclaimer
    };
    console.log('State to save:', JSON.stringify(stateToSave, null, 2));
    
    // Validate state before saving
    if (Object.keys(selectedAnswers).length > 0 || 
        Object.keys(submitted).length > 0 || 
        score > 0 || 
        answeredQuestions.size > 0) {
      localStorage.setItem(`quiz-state-${sessionId}`, JSON.stringify(stateToSave));
      console.log('State saved successfully');
    } else {
      console.log('No meaningful state to save');
    }
  }, [currentQuestionIndex, selectedAnswers, submitted, score, answeredQuestions, showDisclaimer, sessionId]);

  // Clear state when component unmounts
  useEffect(() => {
    return () => {
      console.log('Component unmounting, session:', sessionId);
    };
  }, [sessionId]);

  const currentQuestion = questions[currentQuestionIndex];
  const selectedAnswersForCurrentQuestion = selectedAnswers[currentQuestionIndex] || [];
  const hasSubmitted = submitted[currentQuestionIndex] || false;
  const isDisclaimerVisible = showDisclaimer[currentQuestionIndex] || false;

  const handleAnswerSelect = (answer: string) => {
    if (hasSubmitted) return;

    setSelectedAnswers(prev => {
      const currentAnswers = prev[currentQuestionIndex] || [];
      
      if (currentQuestion.isMultipleAnswer) {
        // For multiple answer questions (checkbox)
        const isSelected = currentAnswers.includes(answer);
        if (isSelected) {
          return {
            ...prev,
            [currentQuestionIndex]: currentAnswers.filter(a => a !== answer)
          };
        } else if (currentAnswers.length < 2) {
          return {
            ...prev,
            [currentQuestionIndex]: [...currentAnswers, answer]
          };
        }
        return prev;
      } else {
        // For single answer questions (radio)
        return {
      ...prev,
          [currentQuestionIndex]: [answer]
        };
      }
    });
  };

  const handleSubmit = () => {
    if (selectedAnswersForCurrentQuestion.length === 0) return;
    setSubmitted(prev => ({ ...prev, [currentQuestionIndex]: true }));
    const isCorrect = currentQuestion.isMultipleAnswer
      ? selectedAnswersForCurrentQuestion.length === currentQuestion.answer.length &&
        selectedAnswersForCurrentQuestion.every(a => currentQuestion.answer.includes(a))
      : selectedAnswersForCurrentQuestion[0] === currentQuestion.answer[0];

    if (isCorrect) {
      setScore(score + 1);
    }
    setAnsweredQuestions(new Map(answeredQuestions).set(currentQuestionIndex, {
      selectedAnswers: selectedAnswersForCurrentQuestion,
      isCorrect
    }));
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePreviousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleToggleDisclaimer = () => {
    setShowDisclaimer(prev => ({
      ...prev,
      [currentQuestionIndex]: !prev[currentQuestionIndex]
    }));
  };

  const handleReset = () => {
    setCurrentQuestionIndex(0);
    setSelectedAnswers({});
    setSubmitted({});
    setShowDisclaimer({});
    setScore(0);
    setAnsweredQuestions(new Map());
  };

  const renderExplanation = (explanation: string) => {
    // Split the explanation into main content and resource URLs
    const [mainContent, sourceUrls] = explanation.split('Resource URL:');

    // Function to extract clean URLs from text
    const extractUrls = (text: string): string[] => {
      if (!text) return [];
      
      // Match all URLs, being careful not to include HTML or trailing characters
      const urlPattern = /https?:\/\/[^\s<>"]+/g;
      const matches = text.match(urlPattern) || [];
      
      // Clean and deduplicate URLs
      const cleanUrls = matches.map(url => {
        // Remove any trailing HTML or punctuation
        return url.replace(/[<>"]/g, '').replace(/[.,;]$/, '').trim();
      });
      
      // Remove duplicates and empty strings
      return [...new Set(cleanUrls)].filter(url => url);
    };

    return (
      <div>
        <div dangerouslySetInnerHTML={{ __html: mainContent }} />
        
        {sourceUrls && (
          <div className="resource-urls" style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #d1d5db' }}>
            <strong>Resource URLs:</strong>
            <ul style={{ listStyle: 'none', padding: 0, margin: '0.5rem 0' }}>
              {extractUrls(sourceUrls).map((url, index) => (
                <li key={index} style={{ margin: '0.25rem 0' }}>
                  <a 
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {url}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="quiz-container">
      <div className="quiz-header">
        <h2>AI Quiz Demo</h2>
        <div className="quiz-controls">
          <span className="learner-id">Learner: {sessionId.replace('learner-', '')}</span>
          <span className="score">Score: {score}/{questions.length}</span>
          <Button variant="normal" onClick={handleReset}>Reset Quiz</Button>
          <Button variant="normal" onClick={() => handleToggleDisclaimer()}>Disclaimer</Button>
        </div>
      </div>

      {isDisclaimerVisible && (
        <div className="disclaimer">
          <h3>Disclaimer</h3>
          <p>This quiz is created solely for demonstration and educational purposes. It is not intended for certification preparation. Some questions and answers may have been generated using AI and should be verified against the official AWS documentation. This is not an official AWS resource and is meant for internal use only.</p>
          <Button onClick={() => handleToggleDisclaimer()}>Accept</Button>
        </div>
      )}

      <div className="question-container">
        <div className="question-header">
          <span className="question-number">Question {currentQuestionIndex + 1} of {questions.length}</span>
          <Button 
            variant="primary" 
            onClick={handleSubmit} 
            disabled={!selectedAnswersForCurrentQuestion.length || hasSubmitted}
          >
            Submit Answer
          </Button>
        </div>

        <div className="question-content">
          <p className="question-text">
            {currentQuestion.question}
            {currentQuestion.isMultipleAnswer && (
              <span className="multiple-answer-indicator"> (Select 2 answers)</span>
            )}
          </p>
          
          <div className="options-container">
            {currentQuestion.options.map((option) => {
              const isCorrect = currentQuestion.answer.includes(option.label);
              const isSelected = selectedAnswersForCurrentQuestion.includes(option.label);

              let optionClass = 'option';
              if (hasSubmitted) {
                if (isCorrect) {
                  // Always show correct answers in green
                  optionClass += ' correct';
                  // If this correct answer was selected, add selected-correct class
                  if (isSelected) {
                    optionClass += ' selected-correct';
                  }
                } else if (isSelected) {
                  // Show incorrect selections in red
                  optionClass += ' incorrect';
                }
              } else if (isSelected) {
                optionClass += ' selected';
              }

              const isDisabled = hasSubmitted || 
                (!isSelected && currentQuestion.isMultipleAnswer && selectedAnswersForCurrentQuestion.length >= 2);

              return (
                <div
                key={option.label}
                  className={optionClass}
                  onClick={() => !isDisabled && handleAnswerSelect(option.label)}
                >
                  <input
                    type={currentQuestion.isMultipleAnswer ? "checkbox" : "radio"}
                    id={`option-${option.label}`}
                    name="answer"
                value={option.label}
                    checked={isSelected}
                    onChange={() => !isDisabled && handleAnswerSelect(option.label)}
                    disabled={isDisabled}
                  />
                  <label htmlFor={`option-${option.label}`}>
                    <span className="option-label">{option.label}.</span>
                    <span className="option-text">{option.text}</span>
                    {hasSubmitted && isCorrect && (
                      <span className="correct-indicator"> ✓</span>
                    )}
                  </label>
                </div>
              );
            })}
          </div>

          {hasSubmitted && (
            <div className="explanation">
              <h3>Explanation</h3>
              {currentQuestion.isMultipleAnswer && (
                <div className="answer-summary">
                  <p>Correct answers: {currentQuestion.answer.sort().join(', ')}</p>
                  <p>Your answers: {selectedAnswersForCurrentQuestion.sort().join(', ')}</p>
                  <p>Score: {selectedAnswersForCurrentQuestion.length === currentQuestion.answer.length && 
                           selectedAnswersForCurrentQuestion.every(a => currentQuestion.answer.includes(a)) ? '1' : '0'} point</p>
                </div>
              )}
              {renderExplanation(currentQuestion.explanation)}
            </div>
          )}
        </div>

        <div className="navigation-buttons">
          <Button
            variant="normal"
            onClick={handlePreviousQuestion}
            disabled={currentQuestionIndex === 0}
          >
            Previous
          </Button>
          <Button
            variant="normal"
            onClick={handleNextQuestion}
            disabled={currentQuestionIndex === questions.length - 1}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
} 