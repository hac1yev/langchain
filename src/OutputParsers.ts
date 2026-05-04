import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';

const ai = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0.7,
});

async function stringParser() {
    const prompt = ChatPromptTemplate.fromTemplate(
        'List: {product_name}' // Syntax must be exactly like this
    );
    
    const parser = new StringOutputParser();
    const chain = prompt.pipe(ai).pipe(parser);
    const response = await chain.invoke({
        product_name: 'Top 10 football club'
    })
    
    console.log(response);
}

async function commaSeparatedParser() {
    const prompt = ChatPromptTemplate.fromTemplate(
        'Provide top 5, separated by commas, for: {product_name}' // Syntax must be exactly like this
    );
    
    const parser = new CommaSeparatedListOutputParser();
    const chain = prompt.pipe(ai).pipe(parser);
    const response = await chain.invoke({
        product_name: 'football club'
    })
    
    console.log(response);
}

async function structuredParser() {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        Extract information from following phrase.
        Formatting instructions: {format_instructions}
        Phrase: {phrase}
    `);

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: 'the name of the person',
        likes: 'what the person likes'
    });

    const chain = templatePrompt.pipe(ai).pipe(outputParser);

    const result = await chain.invoke({
        phrase: 'John likes Pineapple pizza',
        format_instructions: outputParser.getFormatInstructions()
    });

    console.log(result);
}

structuredParser();
// commaSeparatedParser();
// stringParser();