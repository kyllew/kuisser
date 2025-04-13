import { marked } from 'marked';
import { QuizQuestion } from '../types/quiz';

export function parseMarkdownQuiz(markdown: string): QuizQuestion[] {
  const questions: QuizQuestion[] = [];
  const skippedQuestions: { number: string; reason: string }[] = [];
  const expectedRanges = [
    { start: 1, end: 50 },
    { start: 101, end: 120 }
  ];
  
  console.log('Starting to parse markdown...');

  // Clean up the markdown text
  const cleanedMarkdown = markdown
    .replace(/\r\n/g, '\n') // Normalize line endings
    .replace(/\t/g, '    '); // Replace tabs with spaces

  // Split by question markers (Q-prefixed numbers)
  const sections = cleanedMarkdown.split(/(?=^Q\d+)/m);
  console.log(`Found ${sections.length} potential question sections`);

  // Keep track of found question numbers
  const foundQuestionNumbers = new Set<number>();

  for (let i = 0; i < sections.length; i++) {
    const section = sections[i].trim();
    if (!section) continue;

    const lines = section.split('\n');
    console.log(`\n=== Processing section ${i} with ${lines.length} lines ===`);
    console.log('First line:', lines[0]);

    // Get the question number
    const questionNumberMatch = lines[0].match(/^Q(\d+)/);
    if (!questionNumberMatch) {
      console.log(`No question number found in section ${i}`);
      continue;
    }
    const questionNumber = parseInt(questionNumberMatch[1]);
    foundQuestionNumbers.add(questionNumber);
    console.log(`Processing Question ${questionNumber}`);

    // Get the question text - now handling multiple lines until we hit Options section
    let questionText = '';
    let questionLines = [];
    let j = 1;
    while (j < lines.length && !lines[j].toLowerCase().startsWith('options')) {
      const line = lines[j].trim();
      if (line) {
        questionLines.push(line);
      }
      j++;
    }
    questionText = questionLines.join(' ');

    if (!questionText) {
      console.log(`No question text found in section ${i}`);
      continue;
    }

    const question: QuizQuestion = {
      question: questionText,
      options: [],
      answer: [],
      isMultipleAnswer: questionText.toLowerCase().includes('choose two') || 
                       questionText.toLowerCase().includes('(choose two)') ||
                       questionText.toLowerCase().includes('(choose 2)') ||
                       questionText.toLowerCase().includes('choose 2') ||
                       questionText.toLowerCase().includes('select two') ||
                       questionText.toLowerCase().includes('(select two)') ||
                       questionText.toLowerCase().includes('select 2') ||
                       questionText.toLowerCase().includes('two answers') ||
                       questionText.toLowerCase().includes('multiple answers'),
      explanation: ''
    };

    let currentSection: 'options' | 'answer' | 'explanation' = 'options';
    let explanationLines: string[] = [];
    let answerFound = false;
    let optionsFound = false;
    let inOptionsSection = false;
    let inExplanationSection = false;
    let optionsCollected = false;

    // Process the rest of the lines
    for (; j < lines.length; j++) {
      const line = lines[j].trim();
      if (!line) continue;

      console.log(`Processing line ${j}:`, line);

      // Section detection with clear boundaries
      if (line.toLowerCase().startsWith('options')) {
        console.log('Found options section');
        currentSection = 'options';
        inOptionsSection = true;
        inExplanationSection = false;
        optionsCollected = false;
        continue;
      } else if (line.toLowerCase().includes('correct answer') && !line.toLowerCase().includes('because') && !inExplanationSection) {
        console.log('Found answer section');
        currentSection = 'answer';
        inOptionsSection = false;
        inExplanationSection = false;
        optionsCollected = true;
        continue;
      } else if (line.toLowerCase().startsWith('explanation')) {
        console.log('Found explanation section');
        currentSection = 'explanation';
        inOptionsSection = false;
        inExplanationSection = true;
        optionsCollected = true;
        continue;
      }

      // Process content based on section
      switch (currentSection) {
        case 'options':
          if (inOptionsSection && !optionsCollected) {
            const optionMatch = line.match(/^\s*([A-E])\.\s*(.+)/);
            if (optionMatch) {
              question.options.push({
                label: optionMatch[1],
                text: optionMatch[2].trim()
              });
              optionsFound = true;
              console.log(`Added option ${optionMatch[1]} to question ${i}`);
            } else if (!line.toLowerCase().includes('options')) {
              // If we hit a line that's not an option and not the options header,
              // we've reached the end of the options section
              optionsCollected = true;
              console.log('End of options section');
            }
          }
          break;
        case 'answer':
          if (!answerFound) {
            // Handle multiple answers with full text descriptions
            const answerLine = line.trim();
            let answers: string[] = [];
            
            // Match answers in formats like "B. Threat detection" or just "B"
            const answerMatch = answerLine.match(/^([A-E])\s*\.?\s*[^;]*/);
            if (answerMatch) {
              answers.push(answerMatch[1]); // Only take the letter part
            }
            
            if (answers.length > 0) {
              // If we already have answers and find more, combine them
              question.answer = [...new Set([...question.answer, ...answers])].sort();
              console.log(`Found answer ${answers[0]} from line: ${answerLine}`);
            }

            // If this is a multiple answer question and we haven't found all answers yet,
            // don't mark answerFound as true
            if (!question.isMultipleAnswer || question.answer.length >= 2) {
              answerFound = true;
              console.log(`Complete answer set: ${question.answer.join(', ')}`);
            }
          }
          break;
        case 'explanation':
          if (inExplanationSection) {
            // Skip any lines that look like options in the explanation
            if (!line.match(/^\s*[A-E]\./) && !line.toLowerCase().includes('explanation')) {
              // Add an empty line before "because:"
              if (line.toLowerCase().includes('because:')) {
                explanationLines.push(''); // Add empty line BEFORE the line with "because:"
                explanationLines.push(line);
              } else {
                explanationLines.push(line);
              }
              console.log('Added explanation line');
            }
          }
          break;
      }
    }

    // Only add the question if it has all required components
    if (optionsFound && answerFound && explanationLines.length > 0) {
      // Convert explanation lines to HTML, preserving the formatting
      question.explanation = marked.parse(explanationLines.join('\n'), { async: false }) as string;
      questions.push(question);
      console.log(`Successfully added Q${questionNumber} with ${question.options.length} options and ${question.answer.length} answers`);
    } else {
      const reasons = [];
      if (!optionsFound) reasons.push('missing options');
      if (!answerFound) reasons.push('missing answer');
      if (explanationLines.length === 0) reasons.push('missing explanation');
      
      skippedQuestions.push({
        number: questionNumber.toString(),
        reason: reasons.join(', ')
      });
      
      console.log(`Q${questionNumber} skipped - missing required components:`, {
        hasOptions: optionsFound,
        hasAnswer: answerFound,
        hasExplanation: explanationLines.length > 0,
        optionsCount: question.options.length,
        explanationLinesCount: explanationLines.length,
        answers: question.answer
      });
    }
  }

  // After processing all sections, check for missing questions
  console.log('\n=== Question Coverage Analysis ===');
  let totalExpected = 0;
  let totalFound = 0;
  let missingQuestions: number[] = [];

  expectedRanges.forEach(range => {
    for (let num = range.start; num <= range.end; num++) {
      totalExpected++;
      if (!foundQuestionNumbers.has(num)) {
        missingQuestions.push(num);
      } else {
        totalFound++;
      }
    }
  });

  console.log(`Total expected questions: ${totalExpected}`);
  console.log(`Total found questions: ${totalFound}`);
  
  if (missingQuestions.length > 0) {
    console.log('\nMissing questions:');
    missingQuestions.forEach(num => {
      console.log(`- Q${num}`);
    });
  } else {
    console.log('\nAll expected questions were found!');
  }

  if (skippedQuestions.length > 0) {
    console.log('\nSkipped Questions (format issues):');
    skippedQuestions.forEach(q => {
      console.log(`- Q${q.number}: ${q.reason}`);
    });
  }

  // Log any unexpected question numbers (outside our expected ranges)
  const unexpectedQuestions = Array.from(foundQuestionNumbers).filter(num => {
    return !expectedRanges.some(range => num >= range.start && num <= range.end);
  });
  
  if (unexpectedQuestions.length > 0) {
    console.log('\nUnexpected question numbers found:');
    unexpectedQuestions.forEach(num => {
      console.log(`- Q${num}`);
    });
  }

  return questions;
} 