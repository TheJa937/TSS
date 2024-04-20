import os
import uuid
from typing import Dict, List, Any, Callable
from openai import OpenAI
from enum import Enum
import fitz

from fastapi import FastAPI, UploadFile, File
from dataclasses import dataclass, field
from dotenv import load_dotenv
import requests
import langchain
from langchain.docstore.document import Document


load_dotenv("./env.env")

ai21_apiKey = os.getenv("AI21_API_KEY")
openai_apiKey = os.getenv("OPENAI_API_KEY")
app = FastAPI()
client = OpenAI(
    api_key=openai_apiKey
)

with open("prompt", "r") as f:
    prompt = f.read()

ADMIN_PASSWORD = "apfel"


class MachineStatus(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


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

class HandoutAssistant:
    def __init__(self) -> None:
        self.current_pdf_path = None
        self.question_data = None

    def process_pdf(self, pdf_path):
        text, page_texts = PDFReader.pdfToText(pdf_path)
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
        # Convert questions_data to a list of Documents
        documents = [Document(page_content=q_data["segmentText"], metadata={"id": q_data["id"], "page_number": q_data["page_number"]}) for q_data in questions_data]

        # Create the FAISS index (vector store) using the langchain.FAISS.from_documents() method
        vector_store = langchain.FAISS.from_documents(documents, self.embedder)

        return vector_store

def generate_response(messages: List[Dict[str, str]]) -> str:
    """
    Generate AI response based on the conversation history.

    Args:
        messages (List[Dict[str, str]]): List of messages in the conversation history.

    Returns:
        str: Generated response from AI model.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion.choices[0].message.content


@dataclass
class Userport:
    """
    Represents a user port. A user port is a way a user can be authenticated to the system.

    Attributes:
        name (str): Name of the port.
        authentication (Function): Authentication function for the port.
    """
    name: str
    authentication: Callable


@dataclass
class User:
    """
    Represents a user. A user can have multiple ports for authentication.

    Attributes:
        name (str): Name of the user.
        phoneNumber (str): Phone number of the user.
        email (str): Email of the user.
        Userports (list[Userport]): List of user ports.
        currentSessionId (str): Current session ID of the user.
    """
    name: str
    phoneNumber: str
    email: str
    Userports: list[Userport] = field(default_factory=list)
    currentSessionId: str = ""

    def addPhonePort(self):
        """
        Add phone port for the user.
        """
        self.Userports.append(Userport("Phone", lambda number: self.phoneNumber == number))
        Users.append(self)

    def authenticate(self, method, arg):
        """
        Authenticate the user. using the method declared by method

        Args:
            method: Authentication method.
            arg: Argument for authentication.

        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        for port in self.Userports:
            if port.name == method:
                return port.authentication(arg)
        return False


class SessionState(Enum):
    WaitingForServicetechnician = "Waiting for Servicetechnician"
    AwaitingUserResponse = "Awaiting user response"
    PleaseScheduleService = "Please schedule service"
    Closed = "Closed"

    def __str__(self):
        emojies = {self.WaitingForServicetechnician: "🟡",
                   self.AwaitingUserResponse: "🔵",
                   self.PleaseScheduleService: "🔴"}
        return self.value + emojies.get(self, "")


@dataclass
class Session:
    """
    Represents a session. So a problem of a user on a certain machine

    Attributes:
        username (User): User associated with the session.
        id (str): Session ID.
        machine (Machine): Machine associated with the session.
        problem (Problem): Problem associated with the session.
        messages (List[dict[str, str]]): List of messages in the session.
    """
    username: str
    id: str
    machine: "Machine"
    problem: "Problem"
    state: SessionState = SessionState.WaitingForServicetechnician
    messages: List[dict[str, str]] = field(default_factory=list)


@dataclass()
class Machine:
    name: str
    status: MachineStatus = MachineStatus.GREEN


@dataclass
class Problem:
    name: str
    description: str


Sessions = {}
closedSessions = {}
Machines = {"machine1": Machine("machine1"), "machine2": Machine("machine2"), "machine3": Machine("machine3")}
Users = [User("John", "1234567890", "john@johnsmail")]
Users[0].addPhonePort()



def getUsersCurrentSession(username: str):
    """
    Get the current session of the user.

    Args:
        username (str): Username.

    Returns:
        str: Session ID of the user.
    """
    for user in Users:
        if user.name == username:
            return user.currentSessionId
    return None


def authenticateUser(username: str, method: str, arg: Any):
    """
    Authenticate the user.

    Args:
        username (str): Username.
        method (str): Authentication method.
        arg (Any): Argument for authentication.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    for user in Users:
        if user.name == username:
            return user.authenticate(method, arg)
    return False


@dataclass
class AiFunction:
    """
    Represents an AI function.

    Attributes:
        name (str): Name of the AI function.
        function (Function): Function associated with the AI function.
    """
    name: str
    function: Callable


aiFunctions = []


class Specials(Enum):
    Button = "Button"
@dataclass
class Response:
    message: str
    specials: {str: str}


def listSessionStates(context, _) -> list[Response]:
    """
    List session states.

    Args:
        _ : Unused argument.

    Returns:
        dict[str, str]: Session states.
    """
    responses = []
    i = 1
    for session in Sessions.values():
        if session.username == context["user"].name:
            response = Response(f"{session.machine.name}: {session.state}", {Specials.Button: f"Switch to {i}"})
            responses.append(response)
            i += 1
    return responses


aiFunctions.append(AiFunction("listSessionStates", listSessionStates))


def parseAiFunction(call: str, context):
    """
    Parse AI function call.

    Args:
        call (str): AI function call.

    Returns:
        Any: Result of the AI function call.
    """
    if len(call.split("(")) != 2:
        return call
    functionName, arg = call.split("(")
    arg = arg[:-1]
    for function in aiFunctions:
        if function.name == functionName:
            return function.function(context, arg)


@app.get("/")
def test():
    """
    Test API endpoint.
    """
    return {"Hello": "World"}


@app.post("/setMachineStatus")
def setMachineStatus(session_id: str, status: str) -> dict[str, str]:
    """
    Set machine status.

    Args:
        session_id (str): Session ID.
        status (str): Status.

    Returns:
        dict[str, str]: Response message.
    """
    if session_id in Sessions:
        session = Sessions[session_id]
        if status not in MachineStatus.__members__:
            return {"error": "Invalid status"}
        session.machine.status = MachineStatus[status]
        return {"message": "Status updated"}
    return {"error": "Invalid Session ID"}


@app.post("/createProblemSession")
def createProblemSession(username: str, problem: "Problem", machineName: str, authMethod: str, arg) -> dict[str, str]:
    """
    Create a problem session.

    Args:
        username (str): Username.
        problem (Problem): Problem details.
        machineName (str): Machine name.
        authMethod (str): Authentication method.
        arg : Authentication argument.

    Returns:
        dict[str, str]: Response message.
    """
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                session_id = str(uuid.uuid4())
                Sessions[session_id] = Session(session_id, username, Machines[machineName], problem)
                Sessions[session_id].messages.append({"role": "system", "content": prompt})
                user.currentSessionId = session_id
                return {"session_id": session_id}
    return {"error": "Invalid credentials"}


@app.get("/getUserSessions")
def getUserSessions(username: str, authMethod: str, arg) -> dict[str, list[Session]] | dict[str, str]:
    """
    Get user sessions.

    Args:
        username (str): Username.
        authMethod (str): Authentication method.
        arg : Authentication argument.

    Returns:
        dict[str, list[Session]] | dict[str, str]: User sessions or error message.
    """
    for user in Users:
        if user.name == username:
            if user.authenticate(authMethod, arg):
                for session in Sessions.values():
                    print(session)
                return {"sessions": [session for session in Sessions.values() if session.username == username]}
    return {"error": "Invalid credentials"}


@app.post("/closeSession")
def closeSession(session_id: str) -> dict[str, str]:
    """
    Close a session.

    Args:
        session_id (str): Session ID.

    Returns:
        dict[str, str]: Response message.
    """
    if session_id in Sessions:
        closedSessions[session_id] = Sessions.pop(session_id)
        return {"message": "Closed"}
    return {"error": "Invalid session id"}


@app.post("/sendMessage")
def sendMessage(username: str, message: str, authMethod: str, arg: str) -> dict[str, str] | dict[str, list[Any]]:
    """
    Send a message.

    Args:
        username (str): Username.
        message (str): Message content.

    Returns:
        dict[str, str] | dict[str, dict]: Response message or AI generated response.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    sessionId = getUsersCurrentSession(username)
    if sessionId in Sessions:
        session = Sessions[sessionId]
        session.messages.append({"role": "user", "content": message})
        context = {"user": Users[0]}
        response = parseAiFunction(generate_response(session.messages), context)
        return {"message": response}
    return {"error": "Invalid session id"}


@app.get("/getMessages")
def getMessages(username: str, authMethod: str, arg) -> dict[str, Session] | dict[str, str]:
    """
    Get messages.

    Args:
        username (str): Username.

    Returns:
        dict[str, Session] | dict[str, str]: Messages or error message.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        return {"messages": Sessions[session_id].messages}
    return {"error": "Invalid session id"}


@app.get("/getCurrentSession")
def getCurrentSession(username: str, authMethod: str, arg) -> Session | dict[str, str]:
    """
    Get current session.

    Args:
        username (str): Username.

    Returns:
        Session: Current session.
    """
    if not authenticateUser(username, authMethod, arg):
        return {"error": "Invalid credentials"}
    session_id = getUsersCurrentSession(username)
    if session_id in Sessions:
        session = Sessions[session_id]
        return session
    return {"error": "Invalid session id"}

@app.post("/askPDF")
def askPDF(questionData: str) -> str :
    pdf_path = "UploadedFiles/TRUMPF_TruBend_Brochure.pdf"
    handout_assistant = HandoutAssistant()


@app.get("/allActiveSessions")
def getAllActiveSessions(adminPassword: str) -> List[Session] | dict[str, str]:
    if adminPassword == ADMIN_PASSWORD:
        return Sessions
    return {"error": "Invalid Admin Password"}


@app.post("/upload/")
async def uploadfile(file: UploadFile):
    try:
        file_path = f"./UploadedFiles/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}

if __name__ == "__main__":
    import uvicorn

<<<<<<< HEAD
=======
    Sessions["1"] = Session("John", "1", Machines["machine1"], Problem("Problem1", "This is a problem"))
    Sessions["1"].messages.append({"role": "system", "content": prompt})
    Sessions["2"] = Session("John", "2", Machines["machine2"], Problem("Problem2", "This is another problem"))
    Sessions["2"].messages.append({"role": "system", "content": prompt})
    Sessions["2"].state = SessionState.AwaitingUserResponse

    Users[0].currentSessionId = "1"

    pdf_path = "TRUMPF_TruBend_Brochure.pdf"
>>>>>>> 60104c6e9dc59dbd757b74529fd8fbc66a2ceb86
    uvicorn.run(app, host="localhost", port=8000)
