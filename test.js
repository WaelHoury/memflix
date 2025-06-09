const Memflix = require('./memflix');

async function debugTest() {
    console.log('🎬 Debug Testing Memflix...');

    const memflix = new Memflix({
        useOllama: false,
        maxChunkSize: 200 // Smaller chunks for testing
    });

    const knowledge = `
    Artificial intelligence is transforming industries worldwide. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to solve complex problems.
    
    Natural language processing enables computers to understand and generate human language. Computer vision allows machines to interpret and analyze visual information from images and videos.
    
    The future of AI includes autonomous vehicles, medical diagnosis systems, and smart cities that can optimize resource usage and improve quality of life for residents.
    `;

    try {
        console.log('📝 Processing text...');

        // Debug chunking
        const chunks = memflix.chunkText(knowledge);
        console.log(`📊 Chunks created: ${chunks.length}`);
        chunks.forEach((chunk, i) => {
            console.log(`   Chunk ${i + 1}: ${chunk.substring(0, 80)}...`);
        });

        await memflix.processText(knowledge, { source: 'ai_knowledge' });
        console.log(`📊 Total indexed chunks: ${memflix.index.size}`);
        console.log(`📊 Total embeddings: ${memflix.embeddingIndex.size}`);

        // Debug embeddings before encoding
        console.log('\n🔍 Testing search BEFORE encoding:');
        const beforeResults = await memflix.search('machine learning', 2);
        beforeResults.forEach((r, i) => {
            console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] ${r.text.substring(0, 60)}...`);
        });

        console.log('\n🎥 Encoding to video...');
        const result = await memflix.encodeToVideo('./knowledge.mp4');
        console.log(`✅ Created video: ${result.totalChunks} chunks`);

        console.log('\n🔄 Decoding from video...');
        await memflix.decodeFromVideo('./knowledge.mp4');
        console.log(`📊 Decoded chunks: ${memflix.index.size}`);
        console.log(`📊 Decoded embeddings: ${memflix.embeddingIndex.size}`);

        // Debug embeddings after decoding
        console.log('\n🔍 Testing search AFTER decoding:');
        const afterResults = await memflix.search('machine learning', 2);
        if (afterResults.length === 0) {
            console.log('   No results found');

            // Debug: check if embeddings exist
            console.log('\n🔧 Debug info:');
            console.log(`   Index size: ${memflix.index.size}`);
            console.log(`   Embedding index size: ${memflix.embeddingIndex.size}`);
            console.log(`   Embeddings array length: ${memflix.embeddings ? memflix.embeddings.length : 'null'}`);
            console.log(`   Embedding dimension: ${memflix.config.embeddingDim}`);

            // Check first chunk
            const firstChunkId = Array.from(memflix.index.keys())[0];
            if (firstChunkId) {
                const embedding = memflix.getEmbedding(firstChunkId);
                console.log(`   First chunk embedding exists: ${embedding ? 'yes' : 'no'}`);
                if (embedding) {
                    console.log(`   First chunk embedding length: ${embedding.length}`);
                }
            }
        } else {
            afterResults.forEach((r, i) => {
                console.log(`   ${i + 1}. [${r.similarity.toFixed(3)}] ${r.text.substring(0, 60)}...`);
            });
        }

        console.log('\n🎉 Debug completed!');

    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error.stack);
    }
}

debugTest();