import re
import fitz
import requests
import langchain
from langchain.docstore.document import Document
from enum import Enum
import os
from openai import OpenAI
import openai

ai21_apiKey = os.getenv("AI21_API_KEY")
openai_apiKey = os.getenv("OPENAI_API_KEY")

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
            "Authorization": f"Bearer {os.environ[ai21_apiKey]}"
        }
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            json_response = response.json()
            return json_response.get("segments")
        else:
            print(f"An error occurred: {response.status_code}")
            return None

class OpenAIAPI:
    def __init__(self):
        api_key=openai_apiKey

    def get_answer_and_id(self, prompt):
        response = openai.chat.completions.create(
            engine="gpt-3.5-turbo",
            prompt=prompt
        )
        
        answer_text = response.choices[0].text.strip()
        lines = response.choices[0].text.strip().split('\n')
        answer = lines[0].strip()
        try:
            segment_id = int(re.search(r'<ID: (\d+)>', answer).group(1))
            answer = re.sub(r'<ID: \d+>', '', answer).strip()
        except AttributeError:
            segment_id = None
        return answer, segment_id

class HandoutAssistant:
    def __init__(self) -> None:
        self.current_pdf_path = None
        self.question_data = None
        self.pdf_path = "TRUMPF_TruBend_Brochure.pdf"

    def process_pdf(self):
        text, page_texts = PDFReader.pdfToText(self.pdf_path)
        segmented_text = AI21PDFHandler.segment_text(text)
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
            print(f"Element ID: {segment['id']}, Page Number: {segment['page_number']}")
        return segmented_text

    def build_faiss_index(self, questions_data):
        documents = [Document(page_content=q_data["segmentText"], metadata={"id": q_data["id"], "page_number": q_data["page_number"]}) for q_data in questions_data]
        vector_store = langchain.FAISS.from_documents(documents, self.embedder)
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
                    "score": doc.metadata.get("score", None),
                    "page_number": segment["page_number"]
                })
                # print the score and the element ID
                print(f"Element ID: {segment['id']}, Score: {doc.metadata.get('score', None)}")

        relevant_segments.sort(key=lambda x: x["score"] if x["score"] is not None else float('-inf'), reverse=True)
        return relevant_segments
