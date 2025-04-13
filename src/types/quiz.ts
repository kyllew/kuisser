export interface QuizFile {
  value: string;
  label: string;
  description?: string;
}

export interface Question {
  question: string;
  options: {
    label: string;
    text: string;
  }[];
  answer: string[];
  explanation: string;
  isMultipleAnswer: boolean;
}

export interface QuestionAnswer {
  selectedAnswers: string[];
  isCorrect: boolean;
}

export interface QuizQuestion {
  question: string;
  options: {
    label: string;
    text: string;
  }[];
  answer: string[];
  isMultipleAnswer: boolean;
  explanation: string;
}

export interface QuizState {
  currentQuestionIndex: number;
  questions: QuizQuestion[];
  showAnswer: boolean;
  selectedAnswers: string[];
  score: number;
  answeredQuestions: Set<number>;
} 