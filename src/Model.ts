import { ChatOpenAI } from '@langchain/openai';

const ai = new ChatOpenAI({
    model: 'gpt-4o-mini',
    temperature: 0.8,
    maxTokens: 100,
    // verbose: true  // full of unnecessary words  
});

async function main() {
    // const response1 = await ai.invoke('Give me exactly 5 best football players. Only return a short list without explanation'); 
    // console.log(response1.content);

    // const response2 = await ai.batch(['Hello', 'Give me exactly 5 best football players. Only return a short list without explanation']); 
    // for await(const item of response2) {
    //     console.log(item.content);
    // }

    // const response3 = await ai.stream('Give me exactly 5 best football players. Only return a short list without explanation'); 
    // for await (const chunk of response3) {
    //     console.log(chunk.content);
    // }
}

main();