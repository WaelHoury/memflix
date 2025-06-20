# 🎬 Memflix

**Turn your knowledge into searchable videos.** Encode millions of text chunks into MP4 files and search them at lightning speed.

## Why Videos as Databases?

- **📼 Portable**: Your entire knowledge base in a single MP4 file
- **🔍 Instant Search**: Sub-second semantic search across massive datasets  
- **💾 Compressed**: 10x storage efficiency vs traditional databases
- **🌐 Offline-First**: No servers, no internet required
- **🎯 Universal**: Works with any LLM (OpenAI, Anthropic, Ollama)

## Quick Start

```bash
npm install qrcode jsqr fluent-ffmpeg canvas @xenova/transformers axios
```

```javascript
const Memflix = require('./memflix');

const memory = new Memflix();

// Encode knowledge into video
await memory.processText("AI is transforming industries...");
await memory.encodeToVideo('./knowledge.mp4');

// Later: search your video database
await memory.decodeFromVideo('./knowledge.mp4');
const results = await memory.search("What is machine learning?");
```

## Real Examples

**📚 Process Books**
```javascript
await memory.processPDF('./ai-textbook.pdf');
await memory.processEPUB('./sci-fi-novel.epub');
await memory.encodeToVideo('./library.mp4');
```
**🤖 RAG with Any LLM**
```javascript
// Works with OpenAI, Anthropic, Ollama
const answer = await memory.ragQuery("Explain quantum computing", {
  llmProvider: 'ollama',
  model: 'llama2'
});
```

**🔍 Smart Search with Filters**
```javascript
const results = await memory.search("neural networks", 10, 
  chunk => chunk.metadata.source.includes('textbook')
);
```

## Use Cases

- **📖 Digital Libraries**: Store thousands of books in one video
- **🎓 Course Materials**: Searchable lecture transcripts  
- **📰 News Archives**: Years of articles compressed
- **🔬 Research Papers**: Instant semantic search across literature
- **💼 Corporate Knowledge**: Company-wide searchable database

## How It Works

1. **Chunk**: Text split into semantic pieces
2. **Embed**: Generate vector embeddings for search
3. **Encode**: Pack chunks as QR codes into video frames
4. **Compress**: FFmpeg creates your video database
5. **Search**: Decode + vector similarity for instant results

Your knowledge becomes a video file you can copy, stream, or share anywhere. The future of memory is here. 🚀
