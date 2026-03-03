#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

def load_qa_kb(csv_path: str):
    questions, answers = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            if q and a:
                questions.append(q)
                answers.append(a)
    return questions, answers

class SimilarityQA:
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.q_matrix = self.vectorizer.fit_transform(self.questions)

    def answer(self, user_query: str, threshold: float = 0.35):
        if not user_query.strip():
            return None
        q_vec = self.vectorizer.transform([user_query])
        sims = cosine_similarity(q_vec, self.q_matrix).ravel()
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            return self.answers[best_idx], best_score, self.questions[best_idx]
        return None

# Load KB once at startup
kb_questions, kb_answers = load_qa_kb("cocktail_QA_high_paraphrase.csv")
sim_qa = SimilarityQA(kb_questions, kb_answers)

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands

    # If AIML returns a command, handle it as you already do
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)

            except:
                # Wikipedia failed → try similarity KB instead
                sim_result = sim_qa.answer(userInput, threshold=0.35)

                if sim_result:
                    kb_answer, score, matched_q = sim_result
                    print(kb_answer)
                    # Optional debug:
                    # print(f"(matched: '{matched_q}' score={score:.2f})")
                else:
                    print("Sorry, I do not know that. Be more specific!")

        elif cmd == 99:
            # AIML didn't match -> try similarity KB
            sim_result = sim_qa.answer(userInput, threshold=0.35)
            if sim_result:
                kb_answer, score, matched_q = sim_result
                print(kb_answer)
                # Optional debug (remove for final submission):
                # print(f"(matched: '{matched_q}' score={score:.2f})")
            else:
                print("I did not get that, please try again.")

    else:
        # AIML gave a normal answer -> print it
        # If AIML returned empty, also try similarity
        if not answer.strip():
            sim_result = sim_qa.answer(userInput, threshold=0.35)
            if sim_result:
                kb_answer, score, matched_q = sim_result
                print(kb_answer)
            else:
                print("I did not get that, please try again.")
        else:
            print(answer)