import dspy

class VerifyStatement(dspy.Signature):
    """Check the veracity of the following statement."""
    statement = dspy.InputField(desc="The statement to verify")
    evidence = dspy.InputField(desc="Contextual evidence from the corpus")
    answer = dspy.OutputField(desc="True or False, with justification and citing evidence. If do not have enough information, reply: 'I do not have enough information'")

class VeracityChecker(dspy.Module):
    def __init__(self, retriever, llm):
        super().__init__()
        self.retriever = retriever
        self.llm = dspy.Predict(VerifyStatement, model=llm)
        
    def forward(self, statement):
        # Recuperar evidencia
        evidence_docs = self.retriever.retrieve(statement, top_k=5)
        context = "\n".join([doc.page_content for doc in evidence_docs])
        
        # Generar predicción
        response = self.llm(statement=statement, evidence=context)
        return response

class Retriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
    def retrieve(self, query, top_k=5):
        results = self.vectorstore.similarity_search(query, k=top_k)
        return results
