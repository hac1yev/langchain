import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

const ai = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0.7,
});

async function fromTemplate() {
    const prompt = ChatPromptTemplate.fromTemplate(
        'Write a short description for the following product: {product_name}' // Syntax must be exactly like this
    );


    // this code format the prompt and add value into product_name 
    // const wholePrompt = await prompt.format({
    //     product_name: 'bicycle'
    // })
    // console.log(wholePrompt)


    // creating a chain: connecting the model with the prompt
    const chain = prompt.pipe(ai);
    const response = await chain.invoke({
        product_name: 'bicycle'
    })
    console.log(response.content);
}

async function fromMessage() {
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', 'List all trophies of the football team with its count.'],
        ['human', '{teams}']
    ]);

    const chain = prompt.pipe(ai);

    const response = await chain.invoke({
        teams: 'Man Utd'
    });

    console.log(response.content);
}

fromMessage();
// fromTemplate();