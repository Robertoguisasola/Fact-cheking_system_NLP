import dspy

class VerifyStatement(dspy.Signature):
    """
    Signature for verifying the veracity of a statement using contextual evidence.

    Parameters
    ----------
    statement : str
        The statement to be verified.

    evidence : str
        Retrieved contextual evidence from the knowledge base.

    Returns
    -------
    answer : str
        A verdict ("True", "False", or "I do not have enough information") with justification and evidence citation.
    """
    statement = dspy.InputField(desc="The statement to verify")
    evidence = dspy.InputField(desc="Contextual evidence from the corpus")
    answer = dspy.OutputField(
        desc=(
            "True or False, with justification and citing evidence. "
            "If do not have enough information, reply: 'I do not have enough information'"
        )
    )


class VeracityChecker(dspy.Module):
    """
    Module that verifies the veracity of a statement using a retriever and a language model.

    Parameters
    ----------
    retriever : object
        An object with a `retrieve` method to fetch relevant documents.

    llm : object
        A language model compatible with DSPy's Predict module.

    Methods
    -------
    forward(statement)
        Retrieves relevant evidence and generates a veracity judgment using the LLM.
    """

    def __init__(self, retriever, llm):
        super().__init__()
        self.retriever = retriever
        self.llm = dspy.Predict(VerifyStatement, model=llm)
        
    def forward(self, statement):
        """
        Executes the fact-checking process: retrieves evidence and evaluates the statement.

        Parameters
        ----------
        statement : str
            The input claim to be verified.

        Returns
        -------
        dspy.Prediction
            A DSPy object containing the final answer with justification.
        """
        
        # Retrieving evidences
        evidence_docs = self.retriever.retrieve(statement, top_k=5)
        context = "\n".join([doc.page_content for doc in evidence_docs])

        # Generate predictions
        response = self.llm(statement=statement, evidence=context)
        return response


class Retriever:
    """
    Simple retriever that interfaces with a vector store to perform semantic search.

    Parameters
    ----------
    vectorstore : object
        An object that supports `similarity_search(query, k)` for retrieving relevant documents.

    Methods
    -------
    retrieve(query, top_k=5)
        Returns the top_k most relevant documents for a given query.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
    def retrieve(self, query, top_k=5):
        """
        Retrieves top-k similar documents from the vector store.

        Parameters
        ----------
        query : str
            Query string used for similarity search.

        top_k : int, optional
            Number of top documents to return. Default is 5.

        Returns
        -------
        list
            A list of documents most similar to the query.
        """
        results = self.vectorstore.similarity_search(query, k=top_k)
        return results
