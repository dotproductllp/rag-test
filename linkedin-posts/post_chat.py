import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
load_dotenv()

class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGChatBot:
    def __init__(self):
        self.index_name = os.getenv("INDEX_NAME")

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )

        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})
        
        self.llm = ChatOpenAI(
            model_name="gpt-5.4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.app = self._build_graph()

    def _retrieve_node(self, state: GraphState):
        question = state["question"]
        docs = self.retriever.invoke(question)
        return {"context": docs}

    def _generate_node(self, state: GraphState):
        question = state["question"]
        context_docs = state["context"]
        
        context_str = "\n\n".join(
            f"(Page {doc.metadata.get('page', 'N/A')}) {doc.page_content}"
            for doc in context_docs
        )
        
        
        prompt = f"""You are a helpful AI assistant answering questions about latest news from posts.
Use the following retrieved context to answer the user's question.
If you don't know the answer, simply reply "I DONT KNOW".

Context:
{context_str}

Question: {question}
Helpful Answer:"""
        
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def start_chat(self):

        while True:
            user_input = input("You: ")
            
            inputs = {
                "question": user_input,
            }
            
            print("Thinking...")
            result = self.app.invoke(inputs)
            answer = result["answer"]
            
            print(f"AI: {answer}\n")
            
if __name__ == "__main__":
    RAGChatBot().start_chat()