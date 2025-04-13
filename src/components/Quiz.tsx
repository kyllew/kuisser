import { useState } from 'react';
import { Button } from '@cloudscape-design/components';
import './Quiz.css';
import { Question } from '../types/quiz';

interface QuizProps {
  questions: Question[];
}

export default function Quiz({ questions }: QuizProps) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<{ [key: number]: string[] }>({});
  const [showExplanation, setShowExplanation] = useState<{ [key: number]: boolean }>({});
  const [submitted, setSubmitted] = useState<{ [key: number]: boolean }>({});
  const [score, setScore] = useState(0);
  const [answeredQuestions, setAnsweredQuestions] = useState<Map<number, { selectedAnswers: string[]; isCorrect: boolean }>>(new Map());
  const [showDisclaimer, setShowDisclaimer] = useState<{ [key: number]: boolean }>({});

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
    setShowExplanation(prev => ({ ...prev, [currentQuestionIndex]: true }));
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

  const handleToggleExplanation = () => {
    setShowExplanation(prev => ({
      ...prev,
      [currentQuestionIndex]: !prev[currentQuestionIndex]
    }));
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
    setShowExplanation({});
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
          <span className="score">Score: {score}/{questions.length}</span>
          <Button variant="normal" onClick={handleReset}>Reset Quiz</Button>
          <Button variant="normal" onClick={() => handleToggleDisclaimer()}>Disclaimer</Button>
        </div>
      </div>

      {isDisclaimerVisible && (
        <div className="disclaimer">
          <h3>Disclaimer</h3>
          <p>The quiz is just for demo/education purposes only, not intended for any certification preparation. Some of the questions/answers are generated by AI, always refer to the official documentation and validate the answer</p>
          <Button onClick={() => handleToggleDisclaimer()}>Close</Button>
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