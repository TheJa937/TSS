import re

import dotenv
import fitz
import langchain_community
import requests
import langchain
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os

from langchain_core import vectorstores
from openai import OpenAI
import openai
import pickle

dotenv.load_dotenv("env.env")
ai21_apiKey = os.getenv("AI21_API_KEY")
openai_apiKey = os.getenv("OPENAI_API_KEY")
print(ai21_apiKey)


class PDFReader:
    @staticmethod
    def pdfToText(pdf_path: str):
        doc = fitz.open(pdf_path)
        text = ""
        page_texts = []
        for page in doc:
            page_text = page.get_text("text")
            text += page_text
            page_texts.append({"text": page_text, "page_number": page.number})
        return text, page_texts

    @staticmethod
    def highlight_text(input_pdf, output_pdf, text_to_highlight):
        phrases = text_to_highlight.split('\n')
        with fitz.open(input_pdf) as doc:
            for page in doc:
                for phrase in phrases:
                    areas = page.search_for(phrase)
                    if areas:
                        for area in areas:
                            highlight = page.add_highlight_annot(area)
                            highlight.update()
            doc.save(output_pdf)


class AI21PDFHandler:
    @staticmethod
    def segment_text(text):
        url = "https://api.ai21.com/studio/v1/summarize-by-segment"
        payload = {
            "sourceType": "TEXT",
            "source": text
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {ai21_apiKey}"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            json_response = response.json()
            return json_response.get("segments")
        else:
            print(f"An error occurred: {response.status_code}")
            return None


def get_answer_and_id(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    try:
        segment_id = int(re.search(r'\(ID: (\d+)\)', answer).group(1))
        answer = re.sub(r'\(ID: \d+\)', '', answer).strip()
    except AttributeError:
        segment_id = None
    return answer, segment_id


class HandoutAssistant:
    def __init__(self) -> None:
        self.current_pdf_path = None
        self.question_data = None
        self.pdf_path = "./TruLaser-2030-Pre-Install-Manual.pdf"
        self.embedder = OpenAIEmbeddings()
        self.pdf_name = "".join([s for s in self.pdf_path if s.isalnum()])
        self.questions_data = self.process_pdf()
        if os.path.exists("faiis" + self.pdf_name):
            local_index = FAISS.load_local("faiis" + self.pdf_name, self.embedder, allow_dangerous_deserialization=True)
            self.faiss_index = local_index
        else:
            # Build the FAISS index (vector store)
            print("INDEXING NEW FILE")
            self.faiss_index = self.build_faiss_index(self.questions_data)
            self.faiss_index.save_local(folder_path="faiis" + self.pdf_name)

    def process_pdf(self):
        text, page_texts = PDFReader.pdfToText(self.pdf_path)
        text = text[:40_000]
        segmented_text = AI21PDFHandler.segment_text(text)
        print(segmented_text)
        question_data = self.assign_page_numbers_to_pages(segmented_text, page_texts)
        return question_data

    def assign_page_numbers_to_pages(self, segmented_text, page_texts):
        for idx, segment in enumerate(segmented_text):
            segment_text = segment["segmentText"]
            segment["id"] = idx + 1
            max_overlap = 0
            max_overlap_page_number = None
            for page_text in page_texts:
                overlap = len(set(segment_text.split()).intersection(set(page_text["text"].split())))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_page_number = page_text["page_number"]
            segment["page_number"] = max_overlap_page_number + 1
#           print(f"Element ID: {segment['id']}, Page Number: {segment['page_number']}")
        return segmented_text

    def build_faiss_index(self, questions_data):
        documents = [Document(page_content=q_data["segmentText"],
                              metadata={"id": q_data["id"], "page_number": q_data["page_number"]}) for q_data in
                     questions_data]
        vector_store = FAISS.from_documents(documents, self.embedder)
        return vector_store

    def get_relevant_segments(self, questions_data, user_question, faiss_index):
        retriever = faiss_index.as_retriever()
        retriever.search_kwargs = {"k": 5}
        docs = retriever.get_relevant_documents(user_question)
        relevant_segments = []
        for doc in docs:
            segment_id = doc.metadata["id"]
            segment = next((segment for segment in questions_data if segment["id"] == segment_id), None)
            if segment:
                relevant_segments.append({
                    "id": segment["id"],
                    "segment_text": segment["segmentText"],
                    "page_number": segment["page_number"]
                })
                # print the score and the element ID
                print(f"Element ID: {segment['id']}")

        print("most relevant segment: ", relevant_segments[0])
        return relevant_segments

    def generate_prompt(self, question, relevant_segments):
        with open("promptAsk", "r") as f:
            prompt = f.read()
        prompt += f"""
        ###Question: 
        {question}

        ###Relevant Segments:
        """
        #       print("\n\n")
        for segment in relevant_segments:
            prompt += f'\n{segment["id"]}. "{segment["segment_text"]}"'

        #       print(f"Relevant Element ID: {segment['id']}")  # Add this line to print the relevant element IDs
        return prompt

    def get_answer(self, question):
        # Use the retriever to search for the most relevant segments
        relevant_segments = self.get_relevant_segments(self.questions_data, question, self.faiss_index)

        if not relevant_segments:
            return "I couldn't find enough relevant information to answer your question.", None, None, None

        prompt = self.generate_prompt(question, relevant_segments)
        answer, segment_id = get_answer_and_id(prompt)

        if segment_id is not None:
            segment_data = next((seg for seg in relevant_segments if seg["id"] == segment_id), None)
            segment_text = segment_data["segment_text"] if segment_data else None
            page_number = next(
                (segment["page_number"] for segment in self.questions_data if segment["id"] == segment_id), None)
        else:
            page_number = None
            segment_text = None

        return answer, segment_id, segment_text, page_number


ha = HandoutAssistant()


def askQuestion(question: str) -> str:
    print(question)
    return ha.get_answer(question)


if __name__ == "__main__":
    ha = HandoutAssistant()
    print("HI")
    while True:
        print(ha.get_answer(input()))
