const QRCode = require('qrcode');
const jsQR = require('jsqr');
const ffmpeg = require('fluent-ffmpeg');
const { createCanvas, loadImage } = require('canvas');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class Memflix {
    constructor(options = {}) {
        this.embedder = null;
        this.index = new Map();
        this.embeddings = null;
        this.embeddingIndex = new Map();
        this.useOllama = options.useOllama || false;
        this.config = {
            qrSize: 512,
            fps: 30,
            codec: options.codec || 'libx264',
            crf: options.crf || 23,
            maxChunkSize: 1500,
            embeddingDim: 384,
            batchSize: 100,
            ollamaURL: options.ollamaURL || 'http://localhost:11434',
            embeddingModel: options.embeddingModel || 'nomic-embed-text',
            ...options
        };
    }

    async initialize() {
        if (this.useOllama) {
            try {
                await axios.get(`${this.config.ollamaURL}/api/tags`);
            } catch (error) {
                throw new Error('Ollama not accessible. Ensure it\'s running on ' + this.config.ollamaURL);
            }
        } else {
            if (!this.embedder) {
                const { pipeline } = await import('@xenova/transformers');
                this.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
            }
        }
    }

    async generateEmbedding(text) {
        if (this.useOllama) {
            const response = await axios.post(`${this.config.ollamaURL}/api/embeddings`, {
                model: this.config.embeddingModel,
                prompt: text
            });
            return response.data.embedding;
        } else {
            const result = await this.embedder(text);
            return Array.from(result.data);
        }
    }

    async processText(text, metadata = {}) {
        await this.initialize();
        const chunks = this.chunkText(text);
        const results = [];

        for (let i = 0; i < chunks.length; i += this.config.batchSize) {
            const batch = chunks.slice(i, i + this.config.batchSize);
            const batchResults = await this.processBatch(batch, metadata, i);
            results.push(...batchResults);
        }

        return results;
    }

    async processBatch(chunks, metadata, offset) {
        const results = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const chunkId = this.generateId(chunk + offset + i);
            const chunkData = {
                id: chunkId,
                text: chunk,
                metadata: { ...metadata, index: offset + i }
            };

            this.index.set(chunkId, chunkData);
            const embedding = await this.generateEmbedding(chunk);
            this.storeEmbedding(chunkId, embedding);
            results.push(chunkId);
        }

        return results;
    }

    storeEmbedding(chunkId, embedding) {
        if (!this.embeddings) {
            this.embeddings = new Float32Array(0);
            this.config.embeddingDim = embedding.length;
        }

        const offset = this.embeddings.length;
        const newSize = offset + embedding.length;
        const newEmbeddings = new Float32Array(newSize);
        newEmbeddings.set(this.embeddings);
        newEmbeddings.set(embedding, offset);

        this.embeddings = newEmbeddings;
        this.embeddingIndex.set(chunkId, offset);
    }

    getEmbedding(chunkId) {
        const offset = this.embeddingIndex.get(chunkId);
        if (offset === undefined) return null;
        return this.embeddings.slice(offset, offset + this.config.embeddingDim);
    }

    async search(query, limit = 10, filter = null) {
        await this.initialize();

        const queryVector = await this.generateEmbedding(query);

        const candidates = filter ?
            Array.from(this.index.entries()).filter(([id, chunk]) => filter(chunk)) :
            Array.from(this.index.entries());

        const similarities = candidates.map(([chunkId, chunk]) => {
            const embedding = this.getEmbedding(chunkId);
            const similarity = embedding ? this.cosineSimilarity(queryVector, embedding) : 0;
            return { chunkId, similarity, chunk };
        });

        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit)
            .map(item => ({
                text: item.chunk.text,
                metadata: item.chunk.metadata,
                similarity: item.similarity
            }));
    }

    async encodeToVideo(outputPath, options = {}) {
        const tempDir = await this.createTempDir();
        const frameDir = path.join(tempDir, 'frames');
        await fs.mkdir(frameDir, { recursive: true });

        const embeddingPath = outputPath.replace('.mp4', '.emb');
        await this.saveEmbeddings(embeddingPath);

        let frameIndex = 0;
        for (const [chunkId, chunkData] of this.index) {
            const qrData = JSON.stringify({
                id: chunkId,
                text: chunkData.text,
                metadata: chunkData.metadata
            });

            const framePath = path.join(frameDir, `frame_${frameIndex.toString().padStart(6, '0')}.png`);
            await this.generateQRFrame(qrData, framePath);
            frameIndex++;
        }

        await this.encodeFramesToVideo(frameDir, outputPath, options);
        await this.cleanup(tempDir);

        return { videoPath: outputPath, embeddingPath, totalChunks: this.index.size };
    }

    async decodeFromVideo(videoPath) {
        const tempDir = await this.createTempDir();
        const frameDir = path.join(tempDir, 'frames');
        await fs.mkdir(frameDir, { recursive: true });

        const embeddingPath = videoPath.replace('.mp4', '.emb');
        await this.loadEmbeddings(embeddingPath);

        await this.extractFramesFromVideo(videoPath, frameDir);
        const frameFiles = (await fs.readdir(frameDir)).sort();

        this.index.clear();
        let decodedCount = 0;

        for (const frameFile of frameFiles) {
            if (!frameFile.endsWith('.png')) continue;

            const framePath = path.join(frameDir, frameFile);
            const qrData = await this.decodeQRFrame(framePath);

            if (qrData) {
                try {
                    const parsed = JSON.parse(qrData);
                    this.index.set(parsed.id, parsed);
                    decodedCount++;
                } catch (error) {
                    console.warn(`Failed to parse QR data from ${frameFile}`);
                }
            }
        }

        await this.cleanup(tempDir);
        console.log(`🔧 Decoded ${decodedCount} chunks from ${frameFiles.length} frames`);
        return { totalChunks: this.index.size };
    }

    async saveEmbeddings(path) {
        const data = {
            embeddings: Array.from(this.embeddings),
            index: Object.fromEntries(this.embeddingIndex),
            embeddingDim: this.config.embeddingDim,
            useOllama: this.useOllama
        };
        await fs.writeFile(path, JSON.stringify(data));
    }

    async loadEmbeddings(path) {
        const data = JSON.parse(await fs.readFile(path, 'utf8'));
        this.embeddings = new Float32Array(data.embeddings);
        this.embeddingIndex = new Map(Object.entries(data.index).map(([k, v]) => [k, parseInt(v)]));
        this.config.embeddingDim = data.embeddingDim;
    }

    chunkText(text, maxSize = this.config.maxChunkSize) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim());
        const chunks = [];
        let currentChunk = '';

        for (const sentence of sentences) {
            if (currentChunk.length + sentence.length > maxSize && currentChunk) {
                chunks.push(currentChunk.trim());
                currentChunk = sentence;
            } else {
                currentChunk += (currentChunk ? '. ' : '') + sentence;
            }
        }

        if (currentChunk) chunks.push(currentChunk.trim());
        return chunks;
    }

    generateId(text) {
        return crypto.createHash('md5').update(text).digest('hex').substring(0, 16);
    }

    async generateQRFrame(data, outputPath) {
        const canvas = createCanvas(this.config.qrSize, this.config.qrSize);
        const ctx = canvas.getContext('2d');

        const qrDataURL = await QRCode.toDataURL(data, {
            width: this.config.qrSize,
            margin: 2,
            color: { dark: '#000000', light: '#FFFFFF' }
        });

        const img = await loadImage(qrDataURL);
        ctx.drawImage(img, 0, 0);

        const buffer = canvas.toBuffer('image/png');
        await fs.writeFile(outputPath, buffer);
    }

    async decodeQRFrame(framePath) {
        try {
            const img = await loadImage(framePath);
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(0, 0, img.width, img.height);
            const code = jsQR(imageData.data, img.width, img.height);

            if (!code) {
                console.warn(`🔧 No QR code found in ${path.basename(framePath)}`);
            }

            return code ? code.data : null;
        } catch (error) {
            console.warn(`🔧 Error decoding ${path.basename(framePath)}: ${error.message}`);
            return null;
        }
    }

    async encodeFramesToVideo(frameDir, outputPath) {
        return new Promise((resolve, reject) => {
            ffmpeg()
                .input(path.join(frameDir, 'frame_%06d.png'))
                .inputFPS(this.config.fps)
                .videoCodec(this.config.codec)
                .outputOptions(['-crf', this.config.crf.toString(), '-preset', 'medium', '-pix_fmt', 'yuv420p'])
                .output(outputPath)
                .on('end', resolve)
                .on('error', reject)
                .run();
        });
    }

    async extractFramesFromVideo(videoPath, outputDir) {
        return new Promise((resolve, reject) => {
            ffmpeg(videoPath)
                .output(path.join(outputDir, 'frame_%06d.png'))
                .outputOptions(['-vsync', '0'])
                .on('end', resolve)
                .on('error', reject)
                .run();
        });
    }

    // Add these methods to Memflix class:

    async processPDF(pdfPath, metadata = {}) {
        const pdfParse = require('pdf-parse');
        const buffer = await fs.readFile(pdfPath);
        const data = await pdfParse(buffer);
        return this.processText(data.text, {
            ...metadata,
            source: pdfPath,
            type: 'pdf',
            pages: data.numpages
        });
    }

    async processEPUB(epubPath, metadata = {}) {
        const EPub = require('epub2').EPub;

        return new Promise((resolve, reject) => {
            const epub = new EPub(epubPath);
            epub.on('end', async () => {
                const chapters = epub.flow.map(chapter => chapter.id);
                let fullText = '';

                for (const chapterId of chapters) {
                    try {
                        const chapterText = await new Promise((res, rej) => {
                            epub.getChapter(chapterId, (error, text) => {
                                if (error) rej(error);
                                else res(text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' '));
                            });
                        });
                        fullText += chapterText + '\n\n';
                    } catch (error) {
                        console.warn(`Failed to parse chapter ${chapterId}`);
                    }
                }

                try {
                    const result = await this.processText(fullText, {
                        ...metadata,
                        source: epubPath,
                        type: 'epub',
                        title: epub.metadata.title,
                        author: epub.metadata.creator
                    });
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            });

            epub.on('error', reject);
            epub.parse();
        });
    }

    cosineSimilarity(a, b) {
        const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
        const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    async createTempDir() {
        const tempDir = path.join(process.cwd(), 'temp', Date.now().toString());
        await fs.mkdir(tempDir, { recursive: true });
        return tempDir;
    }

    async cleanup(dir) {
        await fs.rm(dir, { recursive: true, force: true });
    }
}

module.exports = Memflix;