// Import vector store (in-memory), prompt builder, LLM, embeddings, and document structure
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "langchain";

// Initialize the LLM (chat model) with temperature for response creativity
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0.7,
});

// Sample data that will be embedded and stored in the vector database
const myData = [
  "My name is John.",
  "My name is Bob.",
  "My favorite food is pizza.",
  "My favorite food is pasta.",
];

// User question we want to answer using RAG (Retrieval-Augmented Generation)
const question = "What are my favorite foods?";

async function main() {
  // Create embedding model to convert text into vectors
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  // Initialize in-memory vector store
  const vectorStore = new MemoryVectorStore(embeddings);
  
  // Convert raw text into Document objects and store them with embeddings
  await vectorStore.addDocuments(
    myData.map((content, index) => {
      return new Document({
        pageContent: content,     // actual text
        metadata: { id: index },  // optional metadata
      });
    }),
  );

  // Create retriever to fetch top-2 most relevant documents
  const retriever = vectorStore.asRetriever({
    k: 2 // return top 2 most similar documents
  });

  // Retrieve relevant documents based on the question
  const result = await retriever._getRelevantDocuments(question);

  // Extract only text content from retrieved documents
  const resultDocuments = result.map(result => result.pageContent);

  // Create prompt template combining system instructions + user input
  const template = ChatPromptTemplate.fromMessages([
    ['system', 'Answer the users question based on the following context: {context}'],
    ['human', '{input}']
  ]);

  // Chain prompt template with the model
  const chain = template.pipe(model);

  // Invoke LLM with question + retrieved context
  const response = await chain.invoke({
    input: question,
    context: resultDocuments
  });

  // Output final generated answer
  console.log(response.content);
}

main();