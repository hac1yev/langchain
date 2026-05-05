// Import vector store (in-memory), prompt builder, LLM, embeddings, and document structure
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "langchain";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";

// Initialize the LLM (chat model) with temperature for response creativity
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0.7,
});

// User question we want to answer using RAG (Retrieval-Augmented Generation)
const question = "What this pdf about";

const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
  ssl: false,
});

async function main() {
  // Create embedding model to convert text into vectors
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  // create a loader
  const loader = new PDFLoader("arayis.pdf", {
    splitPages: false,
  });
  const docs = await loader.load();

  // split the docs
  const splitter = new RecursiveCharacterTextSplitter({
    separators: [". \n"],
  });

  const splittedDocs = await splitter.splitDocuments(docs);

  // Chroma only accepts primitive metadata values, so keep the useful fields
  // and drop nested PDFLoader metadata such as the "pdf" object.
  const chromaDocs = splittedDocs.map((doc, index) => {
    return new Document({
      pageContent: doc.pageContent,
      metadata: {
        id: index,
        source: doc.metadata.source,
      },
    });
  });

  // Initialize Chroma vector store and add PDF chunks
  const vectorStore = await Chroma.fromDocuments(chromaDocs, embeddings, {
    collectionName: "foo",
    index: chromaClient,
  });

  // Create retriever to fetch top-2 most relevant documents
  const retriever = vectorStore.asRetriever({
    k: 2, // return top 2 most similar documents
  });

  // Retrieve relevant documents based on the question
  const result = await retriever.invoke(question);

  // Extract only text content from retrieved documents
  const resultDocuments = result.map((result) => result.pageContent).join("\n\n");

  // Create prompt template combining system instructions + user input
  const template = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the users question based on the following context: {context}",
    ],
    ["human", "{input}"],
  ]);

  // Chain prompt template with the model
  const chain = template.pipe(model);

  // Invoke LLM with question + retrieved context
  const response = await chain.invoke({
    input: question,
    context: resultDocuments,
  });

  // Output final generated answer
  console.log(response.content);
}

main();
