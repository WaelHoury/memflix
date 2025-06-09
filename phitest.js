const Memflix = require('./memflix');

async function testOllamaPhi3() {
    const memflix = new Memflix({
        useOllama: false  // Use local transformers for embeddings
    });

    const knowledge = `
    john doe writes code with the precision of a compiler and the creativity of an artist.  
Backend architectures bend to his will as he builds fast, scalable APIs in Node.js.  
Vue 3 is his brush, MongoDB his canvas—he paints seamless applications with clean UX and real-time performance.  
Artificial intelligence isn't a buzzword; it's a tool he molds into intelligent features and smart systems.  
   `;

    await memflix.processText(knowledge);
    await memflix.encodeToVideo('./phi3-knowledge.mp4');
    await memflix.decodeFromVideo('./phi3-knowledge.mp4');

    const searchResults = await memflix.search('quantum computing');
    console.log('Search Results:', searchResults);

    const ragAnswer = await memflix.ragQuery('what does john doe do?', {
        llmProvider: 'ollama',
        model: 'phi3:mini',
        baseURL: 'http://localhost:11434'
    });

    console.log('RAG Answer:', ragAnswer);
}

testOllamaPhi3().catch(console.error);