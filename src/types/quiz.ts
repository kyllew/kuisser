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