// Thronglet Language Learning System
// Integrates a lightweight language model with the Thronglet neural network

// Global variables
let languageModel = null;
let vocabularyBase = new Map();
let modelLoaded = false;
let isLoading = false;
let languageEnabled = true;
let languageMemory = [];
let wordAssociations = {};
let wordEmbeddings = {}; // Store word embeddings for similarity calculations
let languageLearningRate = 0.05;
const vocabLimit = 1000;            // Maximum vocabulary size
const contextWindow = 5;            // How many previous interactions to remember per Thronglet
// Repetition tracking variables
const recentWordsUsed = new Map();  // Map of throngletId -> array of recent words used
const MAX_RECENT_WORDS = 10;        // How many recent words to track per thronglet
const REPETITION_PENALTY = 0.15;    // Happiness penalty for repetition

// Morse code dictionary
const morseCodeMap = {
    'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.', 
    'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..', 
    'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.', 
    's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-', 
    'y': '-.--', 'z': '--..', 
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', 
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
    ' ': '/'
};

// Reverse mapping for decoding
const reverseMorseCodeMap = {};
Object.keys(morseCodeMap).forEach(key => {
    reverseMorseCodeMap[morseCodeMap[key]] = key;
});

// Fireworks AI API configuration
const FIREWORKS_API_KEY = "fw_3ZLZxEU9XkYj9MdDRyzYfCxn";
const FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions";
const FIREWORKS_MODEL = "accounts/fireworks/models/llama4-maverick-instruct-basic";

// Context length management
const MAX_CONTEXT_LENGTH = 1000000; // 1 million tokens max
let contextLength = 0;

// Auto-teaching system
let isAutoTeachEnabled = false;
let autoTeachInterval = null;
let autoTeachStartTime = null;
const AUTO_TEACH_INITIAL_FREQUENCY = 500; // Start with rapid teaching (500ms)
const AUTO_TEACH_LATER_FREQUENCY = 60000; // Slow down to 1 minute after initial period
const AUTO_TEACH_INITIAL_PERIOD = 600000; // 10 minutes of rapid teaching
let autoTaughtWords = []; // Track words that have been auto-taught
const MAX_AUTO_TAUGHT_HISTORY = 50; // Maximum number of words to remember as "recently taught"

// Initialize language model
async function initializeLanguageModel() {
    if (isLoading || modelLoaded) return;
    
    try {
        isLoading = true;
        
        // Show loading indicator
        const statusLabel = document.createElement('div');
        statusLabel.id = 'language-status';
        statusLabel.textContent = 'Loading language system...';
        statusLabel.style.color = 'white';
        statusLabel.style.backgroundColor = '#8e44ad';
        statusLabel.style.padding = '8px 15px';
        statusLabel.style.borderRadius = '5px';
        statusLabel.style.position = 'absolute';
        statusLabel.style.top = '10px';
        statusLabel.style.left = '50%';
        statusLabel.style.transform = 'translateX(-50%)';
        statusLabel.style.zIndex = '1000';
        
        document.body.appendChild(statusLabel);
        
        // Load TensorFlow.js
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js');
        
        // Load Universal Sentence Encoder (lightweight alternative to BERT)
        await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder@1.3.3/dist/universal-sentence-encoder.min.js');
        
        // Initialize the model
        console.log("Loading Universal Sentence Encoder model...");
        
        // Wait a moment to ensure the script is fully loaded and the global variable is available
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Check if the global variable is available
        if (typeof window.use === 'undefined') {
            console.error("Universal Sentence Encoder module not available");
            throw new Error("Failed to load Universal Sentence Encoder");
        }
        
        // Load the model from the global namespace
        languageModel = await window.use.load();
        console.log("Language model loaded successfully!");
        
        // Initialize seed vocabulary
        initializeVocabulary();
        
        // Update status
        statusLabel.textContent = 'Language system ready!';
        statusLabel.style.backgroundColor = '#27ae60';
        modelLoaded = true;
        
        // Try to load saved state
        setTimeout(() => {
            try {
                if (modelLoaded) {
                    const loaded = loadGameState();
                    if (loaded) {
                        console.log("Successfully loaded saved game state");
                        statusLabel.textContent = 'Game state loaded!';
                    }
                }
            } catch (error) {
                console.error("Error auto-loading game state:", error);
            }
        }, 2000); // Wait 2 seconds after initialization
        
        // Fade out after 5 seconds (extended to allow for state loading)
        setTimeout(() => {
            statusLabel.style.opacity = '0';
            statusLabel.style.transition = 'opacity 1s';
            setTimeout(() => statusLabel.remove(), 1000);
        }, 5000);
        
    } catch (error) {
        console.error("Error loading language model:", error);
        const statusLabel = document.getElementById('language-status');
        if (statusLabel) {
            statusLabel.textContent = 'Error loading language model';
            statusLabel.style.backgroundColor = '#e74c3c';
        }
    } finally {
        isLoading = false;
    }
}

// Helper function to load scripts dynamically
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Initialize the vocabulary with seed words - REDUCED to minimal foundation
function initializeVocabulary() {
    const seedWords = [
        // Absolute minimal seed vocabulary
        { word: "I", category: "pronoun", contexts: ["self"], weight: 0.6 },
        { word: "you", category: "pronoun", contexts: ["other"], weight: 0.6 },
        { word: "good", category: "value", contexts: ["positive"], weight: 0.7 },
        { word: "bad", category: "value", contexts: ["negative"], weight: 0.7 },
        // Basic needs only - will learn the rest through experience
        { word: "food", category: "need", contexts: ["eat"], weight: 0.6 },
        { word: "play", category: "activity", contexts: ["fun"], weight: 0.6 }
    ];
    
    // Add Morse code patterns to all seed words
    seedWords.forEach(word => {
        // Generate Morse code pattern for this word
        word.morsePattern = textToMorse(word.word);
        
        // Store in vocabulary base
        vocabularyBase.set(word.word, word);
    });
    
    console.log("Minimal seed vocabulary initialized with", seedWords.length, "words - rest will emerge naturally");
    
    // Create minimal associations
    createWordAssociations();
}

// Create associations between words - simplified for natural emergence
function createWordAssociations() {
    // Just create the most fundamental associations - rest will emerge through experience
    if (vocabularyBase.has("food")) {
        const foodWord = vocabularyBase.get("food");
        if (!foodWord.associations) foodWord.associations = [];
        foodWord.associations.push({word: "good", strength: 0.6});
    }
    
    if (vocabularyBase.has("play")) {
        const playWord = vocabularyBase.get("play");
        if (!playWord.associations) playWord.associations = [];
        playWord.associations.push({word: "good", strength: 0.6});
    }
}

// Get embedding for text using the language model
async function getEmbedding(text) {
    if (!modelLoaded || !languageModel) return null;
    
    try {
        const embeddings = await languageModel.embed(text);
        const embedding = await embeddings.array();
        return embedding[0]; // Return the first embedding
    } catch (error) {
        console.error("Error getting embedding:", error);
        return null;
    }
}

// Calculate similarity between two texts
async function calculateSimilarity(textA, textB) {
    const embeddingA = await getEmbedding(textA);
    const embeddingB = await getEmbedding(textB);
    
    if (!embeddingA || !embeddingB) return 0;
    
    // Calculate cosine similarity
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < embeddingA.length; i++) {
        dotProduct += embeddingA[i] * embeddingB[i];
        normA += embeddingA[i] * embeddingA[i];
        normB += embeddingB[i] * embeddingB[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Learn a new word and add it to vocabulary
function learnWord(word, context, throngletId = null) {
    word = word.toLowerCase().trim();
    if (!word || word.length < 2) return false;
    
    // Generate Morse code for this word
    const morsePattern = textToMorse(word);
    
    // Track if this is a new word
    let isNewWord = false;
    
    if (vocabularyBase.has(word)) {
        // Word exists, update
        const vocab = vocabularyBase.get(word);
        
        // Add new context if not already present
        if (context && !vocab.contexts.includes(context)) {
            vocab.contexts.push(context);
        }
        
        // Increase weight (learning reinforcement)
        vocab.weight = Math.min(1.0, vocab.weight + 0.05);
        vocab.lastUsed = Date.now();
        
        // Add or update Morse pattern
        if (!vocab.morsePattern) {
            vocab.morsePattern = morsePattern;
        }
        
        return true;
    } else {
        // New word
        isNewWord = true;
        vocabularyBase.set(word, {
            word: word,
            category: "learned",
            contexts: context ? [context] : [],
            weight: 0.3, // Start with low weight
            learnedFrom: throngletId || "creator",
            learned: Date.now(),
            lastUsed: Date.now(),
            associations: [], // Initialize empty associations
            morsePattern: morsePattern // Add Morse code pattern
        });
        
        console.log(`New word learned: "${word}" from ${throngletId ? 'Thronglet #' + throngletId : 'Creator'} with Morse pattern: ${morsePattern}`);
        
        // Activate the word in the Thronglet's brain if it's new
        if (isNewWord && throngletId !== null) {
            // Schedule activation with a small delay to avoid overwhelming the system
            setTimeout(() => {
                activateWordInBrain(word, throngletId);
            }, Math.random() * 2000 + 500); // Random delay between 0.5-2.5 seconds
        }
        
        return true;
    }
}

// Function to have a Thronglet say a word to activate it in their neural network
function activateWordInBrain(word, throngletId) {
    // Find the Thronglet with the given ID
    if (!window.thronglets) return;
    
    const thronglet = window.thronglets.find(t => t.id === parseInt(throngletId, 10));
    if (!thronglet) return;
    
    // Make sure the Thronglet isn't currently speaking
    if (thronglet.isCurrentlySpeaking) return;
    
    // Have the Thronglet say the word with a special thought bubble format
    thronglet.showThought(`🧠 I'm learning the word "${word}"! 🧠`);
    
    console.log(`Thronglet #${throngletId} activated word "${word}" in their neural network`);
    
    // If the Thronglet has an agent, reinforce this concept
    if (thronglet.agent) {
        // Initialize concept knowledge if it doesn't exist
        if (!thronglet.agent.conceptKnowledge) {
            thronglet.agent.conceptKnowledge = {};
        }
        
        // Add or strengthen this word concept
        if (!thronglet.agent.conceptKnowledge[word]) {
            thronglet.agent.conceptKnowledge[word] = 0.1; // Initial concept strength
        } else {
            // Strengthen knowledge each time the word is activated
            thronglet.agent.conceptKnowledge[word] = 
                Math.min(1.0, thronglet.agent.conceptKnowledge[word] + 0.05);
        }
    }
}

// Associate words with each other (like "eat" with "good")
function associateWords(word1, word2, strength = 0.5, context = null) {
    if (!vocabularyBase.has(word1) || !vocabularyBase.has(word2)) {
        return false;
    }
    
    const vocab1 = vocabularyBase.get(word1);
    
    // Initialize associations array if it doesn't exist
    if (!vocab1.associations) vocab1.associations = [];
    
    // Check if association already exists
    const existingAssoc = vocab1.associations.find(a => a.word === word2);
    if (existingAssoc) {
        // Strengthen existing association
        existingAssoc.strength = Math.min(1.0, existingAssoc.strength + 0.1);
        if (context && !existingAssoc.contexts) existingAssoc.contexts = [context];
        else if (context) existingAssoc.contexts.push(context);
    } else {
        // Create new association
        vocab1.associations.push({
            word: word2,
            strength: strength,
            contexts: context ? [context] : []
        });
    }
    
    console.log(`Associated "${word1}" with "${word2}" (strength: ${strength})`);
    return true;
}

// Learn from a sentence (break it into words)
function learnFromSentence(sentence, context, throngletId = null) {
    if (!sentence) return;
    
    // Simple tokenization
    const words = sentence.toLowerCase()
                      .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "")
                      .split(/\s+/);
    
    // Learn individual words
    for (const word of words) {
        if (word.length >= 2) {
            learnWord(word, context, throngletId);
        }
    }
    
    // Learn associations between words (pairs)
    for (let i = 0; i < words.length - 1; i++) {
        if (words[i].length >= 2 && words[i+1].length >= 2) {
            // Check for value associations (good/bad)
            if (words[i+1] === "good" || words[i+1] === "bad") {
                associateWords(words[i], words[i+1], 0.6, context);
                
                // IMPORTANT FIX: Update association in vocabularyBase as well
                if (vocabularyBase.has(words[i]) && vocabularyBase.has(words[i+1])) {
                    const vocabItem = vocabularyBase.get(words[i]);
                    
                    // Initialize associations array if it doesn't exist
                    if (!vocabItem.associations) {
                        vocabItem.associations = [];
                    }
                    
                    // Check if association already exists
                    const existingAssoc = vocabItem.associations.find(a => a.word === words[i+1]);
                    if (existingAssoc) {
                        // Strengthen existing association
                        existingAssoc.strength = Math.min(1.0, existingAssoc.strength + 0.1);
                    } else {
                        // Create new association
                        vocabItem.associations.push({
                            word: words[i+1],
                            strength: 0.6,
                            contexts: context ? [context] : []
                        });
                    }
                    
                    console.log(`Updated vocabularyBase association: "${words[i]}" -> "${words[i+1]}"`);
                }
            }
        }
    }
    
    // Look for specific patterns like "eat good" or "play good"
    if (sentence.toLowerCase().includes("eat good") || sentence.toLowerCase().includes("eating good")) {
        associateWords("food", "good", 0.7, "eating");
        associateWords("apple", "good", 0.7, "eating");
        
        // IMPORTANT FIX: Update these associations in vocabularyBase
        updateVocabularyBaseAssociation("food", "good", 0.7, "eating");
        updateVocabularyBaseAssociation("apple", "good", 0.7, "eating");
    }
    
    if (sentence.toLowerCase().includes("play good") || sentence.toLowerCase().includes("playing good")) {
        associateWords("play", "good", 0.7, "playing");
        associateWords("ball", "good", 0.7, "playing");
        
        // IMPORTANT FIX: Update these associations in vocabularyBase
        updateVocabularyBaseAssociation("play", "good", 0.7, "playing");
        updateVocabularyBaseAssociation("ball", "good", 0.7, "playing");
    }
    
    // Record in language memory
    languageMemory.push({
        sentence: sentence,
        context: context,
        source: throngletId || "creator",
        timestamp: Date.now()
    });
    
    // Limit memory size
    if (languageMemory.length > 100) {
        languageMemory.shift();
    }
    
    return words.length;
}

// IMPORTANT FIX: Helper function to update associations in vocabularyBase
function updateVocabularyBaseAssociation(word1, word2, strength, context) {
    if (vocabularyBase.has(word1) && vocabularyBase.has(word2)) {
        const vocabItem = vocabularyBase.get(word1);
        
        // Initialize associations array if it doesn't exist
        if (!vocabItem.associations) {
            vocabItem.associations = [];
        }
        
        // Check if association already exists
        const existingAssoc = vocabItem.associations.find(a => a.word === word2);
        if (existingAssoc) {
            // Strengthen existing association
            existingAssoc.strength = Math.min(1.0, existingAssoc.strength + 0.1);
        } else {
            // Create new association
            vocabItem.associations.push({
                word: word2,
                strength: strength,
                contexts: context ? [context] : []
            });
        }
        
        console.log(`Updated vocabularyBase association: "${word1}" -> "${word2}"`);
    }
}

// Extract Morse code patterns from text
function extractMorsePatterns(text) {
    // Look for patterns of dots and dashes separated by spaces
    const morseRegex = /[.\-]{1,7}(?:\s+[.\-]{1,7})*/g;
    return text.match(morseRegex) || [];
}

// Convert Morse code to text
function morseToText(morse) {
    const morseMap = {
        ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e",
        "..-.": "f", "--.": "g", "....": "h", "..": "i", ".---": "j",
        "-.-": "k", ".-..": "l", "--": "m", "-.": "n", "---": "o",
        ".--.": "p", "--.-": "q", ".-.": "r", "...": "s", "-": "t",
        "..-": "u", "...-": "v", ".--": "w", "-..-": "x", "-.--": "y",
        "--..": "z", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
        "-----": "0", ".-.-.-": ".", "--..--": ",", "..--..": "?"
    };
    
    const parts = morse.trim().split(/\s+/);
    let result = '';
    
    for (const part of parts) {
        if (morseMap[part]) {
            result += morseMap[part];
        }
    }
    
    return result;
}

// Convert text to Morse code
function textToMorse(text) {
    const morseMap = {
        "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".",
        "f": "..-.", "g": "--.", "h": "....", "i": "..", "j": ".---",
        "k": "-.-", "l": ".-..", "m": "--", "n": "-.", "o": "---",
        "p": ".--.", "q": "--.-", "r": ".-.", "s": "...", "t": "-",
        "u": "..-", "v": "...-", "w": ".--", "x": "-..-", "y": "-.--",
        "z": "--..", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
        "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
        "0": "-----", ".": ".-.-.-", ",": "--..--", "?": "..--.."
    };
    
    text = text.toLowerCase();
    let result = '';
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (morseMap[char]) {
            result += morseMap[char] + ' ';
        }
    }
    
    return result.trim();
}

// Extract known words from input
function extractKnownWords(input) {
    const words = input.toLowerCase().split(/\s+/);
    return words.filter(word => wordAssociations[word]);
}

// Find similar words based on embeddings
async function findSimilarWords(embedding, count = 3) {
    const similarWords = [];
    
    // Calculate cosine similarity with all known words
    const knownWords = Object.keys(wordAssociations);
    const similarities = [];
    
    for (const word of knownWords) {
        if (wordEmbeddings[word]) {
            const similarity = calculateCosineSimilarity(embedding, wordEmbeddings[word]);
            similarities.push({ word, similarity });
        }
    }
    
    // Sort by similarity and get top count
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    // Return the most similar words
    return similarities.slice(0, count).map(item => item.word);
}

// Calculate cosine similarity between two embeddings
function calculateCosineSimilarity(embeddingA, embeddingB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < embeddingA.length; i++) {
        dotProduct += embeddingA[i] * embeddingB[i];
        normA += embeddingA[i] * embeddingA[i];
        normB += embeddingB[i] * embeddingB[i];
    }
    
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    
    return dotProduct / (normA * normB);
}

// Select a random phrase from the given array
function selectRandomPhrase(phrases) {
    return phrases[Math.floor(Math.random() * phrases.length)];
}

// Add function to update the processLanguageInput function to detect and process Morse code
async function processLanguageInput(input, thronglet) {
    if (!modelLoaded || !thronglet || !thronglet.agent) {
        return "I'm still learning...";
    }
    
    try {
        // Check if input contains Morse code
        const morsePatterns = extractMorsePatterns(input);
        
        // If Morse code is detected, process it
        if (morsePatterns.length > 0) {
            // Process each Morse pattern
            for (const pattern of morsePatterns) {
                const decodedText = morseToText(pattern);
                if (decodedText && decodedText.length > 0) {
                    // Learn this word with special Morse context
                    learnFromSentence(decodedText, "morse_communication", thronglet.id);
                    
                    // Show that we understand the Morse code
                    const responseWord = decodedText.toLowerCase();
                    
                    // NEW: Ensure the Morse pattern is saved to vocabulary
                    if (vocabularyBase.has(responseWord)) {
                        const vocabItem = vocabularyBase.get(responseWord);
                        vocabItem.morsePattern = pattern;
                        vocabItem.weight = Math.min(1.0, vocabItem.weight + 0.1);
                    } else {
                        // If the word isn't in vocabulary yet, add it
                        learnWord(responseWord, "morse", thronglet.id);
                        if (vocabularyBase.has(responseWord)) {
                            vocabularyBase.get(responseWord).morsePattern = pattern;
                        }
                    }
                    
                    // NEW: Respond with both text and a different word in Morse
                    // Get a related word if possible
                    let relatedWord = "";
                    
                    if (vocabularyBase.has(responseWord) && 
                        vocabularyBase.get(responseWord).associations && 
                        vocabularyBase.get(responseWord).associations.length > 0) {
                        
                        const assoc = vocabularyBase.get(responseWord).associations[0];
                        relatedWord = assoc.word;
                    } else {
                        // Pick a simple response word
                        const simpleWords = ["yes", "good", "understand", "learn"];
                        relatedWord = simpleWords[Math.floor(Math.random() * simpleWords.length)];
                    }
                    
                    // Get Morse for the related word
                    const relatedMorse = getConceptAsMorse(relatedWord);
                    
                    return `${decodedText}! ${relatedMorse}`;
                }
            }
        }
        
        // Get embedding for input
        const embedding = await getEmbedding(input);
        if (!embedding) return "Beep?";
        
        // Integrate with neural network
        const integration = await integrateWithNeuralNetwork(thronglet.agent, embedding, input);
        
        // Learn from input (if from creator)
        learnFromSentence(input, "creator_input");
        
        // Generate a response based on thronglet's current state and vocabulary
        const response = generateResponse(thronglet, integration);
        
        // Learn from own response
        learnFromSentence(response, "thronglet_output", thronglet.id);
        
        // Check for repetitive word usage
        checkForRepetitiveWords(thronglet, response);
        
        return response;
    } catch (error) {
        console.error("Error processing language:", error);
        return "Beep! Error.";
    }
}

// Function to check for repetitive word usage and apply penalties
function checkForRepetitiveWords(thronglet, response) {
    if (!thronglet || !thronglet.id) return;
    
    // Initialize array for this thronglet if it doesn't exist
    if (!recentWordsUsed.has(thronglet.id)) {
        recentWordsUsed.set(thronglet.id, []);
    }
    
    // Get the words from the response
    const words = response.toLowerCase().split(/\s+/).filter(word => word.length > 2);
    
    // Get recent words for this thronglet
    const recentWords = recentWordsUsed.get(thronglet.id);
    
    // Check for repetitions
    let repetitionFound = false;
    let repetitiveWord = null;
    
    for (const word of words) {
        // Count occurrences of this word in recent words
        const occurrences = recentWords.filter(w => w === word).length;
        
        // If word appears too often, apply a penalty
        if (occurrences >= 2) {
            repetitionFound = true;
            repetitiveWord = word;
            break;
        }
    }
    
    // Apply penalty if repetition found
    if (repetitionFound && thronglet.happiness !== undefined) {
        // Apply a small penalty to happiness
        thronglet.happiness = Math.max(0, thronglet.happiness - REPETITION_PENALTY);
        console.log(`Thronglet #${thronglet.id} penalized for repetitive use of "${repetitiveWord}"`);
        
        // If intelligence is defined, encourage learning new words
        if (thronglet.agent && thronglet.agent.intelligenceScore !== undefined) {
            // Update agent weights to encourage diversity of expression
            const intelligenceBoostChance = 0.3;
            if (Math.random() < intelligenceBoostChance) {
                // Small intelligence boost to encourage learning new words
                thronglet.agent.intelligenceScore += 0.01;
            }
        }
    }
    
    // Update recent words list with words from this response
    for (const word of words) {
        recentWords.push(word);
    }
    
    // Keep only the most recent MAX_RECENT_WORDS
    if (recentWords.length > MAX_RECENT_WORDS) {
        recentWords.splice(0, recentWords.length - MAX_RECENT_WORDS);
    }
    
    // Update the map
    recentWordsUsed.set(thronglet.id, recentWords);
}

// Integrate language embedding with neural network
async function integrateWithNeuralNetwork(agent, embedding, originalText) {
    if (!agent || !embedding) return null;
    
    // Extract emotional dimensions from embedding
    // This is a simplified representation of the semantic content
    const emotionalVector = {
        urgency: 0,  // Related to hunger/needs
        positivity: 0, // Related to happiness/play
        sociality: 0,  // Related to social interaction
        complexity: 0   // Overall linguistic complexity
    };
    
    // Extract semantic features from embedding (approximate)
    // In a real implementation, this would use a more sophisticated technique
    
    // Analyze for urgency (hunger/need related words)
    const urgencyWords = ["hungry", "food", "eat", "need", "help"];
    for (const word of urgencyWords) {
        if (originalText.toLowerCase().includes(word)) {
            emotionalVector.urgency += 0.2;
        }
    }
    
    // Analyze for positivity (happiness/play related words)
    const positivityWords = ["happy", "play", "fun", "good", "like", "ball"];
    for (const word of positivityWords) {
        if (originalText.toLowerCase().includes(word)) {
            emotionalVector.positivity += 0.2;
        }
    }
    
    // Analyze for sociality (social interaction related words)
    const socialityWords = ["friend", "together", "hello", "hi", "we", "us"];
    for (const word of socialityWords) {
        if (originalText.toLowerCase().includes(word)) {
            emotionalVector.sociality += 0.2;
        }
    }
    
    // Calculate rough complexity based on sentence length and vocab diversity
    const words = originalText.split(/\s+/);
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    emotionalVector.complexity = Math.min(1.0, 
        (words.length / 10) * 0.5 + (uniqueWords.size / words.length) * 0.5
    );
    
    // Normalize values between 0 and 1
    Object.keys(emotionalVector).forEach(key => {
        emotionalVector[key] = Math.min(1.0, Math.max(0, emotionalVector[key]));
    });
    
    // Influence neural network weights based on language understanding
    const weightInfluence = {};
    
    // Map emotional dimensions to relevant agent weights
    for (const key in agent.weights) {
        // Initialize with no influence
        weightInfluence[key] = 0;
        
        // Apply specific influences based on weight type
        if (key.includes('hunger') || key.includes('moveToApple')) {
            weightInfluence[key] = emotionalVector.urgency * 0.3;
        }
        else if (key.includes('happiness') || key.includes('moveToBall')) {
            weightInfluence[key] = emotionalVector.positivity * 0.3;
        }
        else if (key.includes('Thronglet') || key.includes('beep')) {
            weightInfluence[key] = emotionalVector.sociality * 0.3;
        }
    }
    
    // Create temporary influenced weights
    const influencedWeights = {};
    for (const key in agent.weights) {
        influencedWeights[key] = agent.weights[key] * (1 + weightInfluence[key]);
    }
    
    return {
        embedding: embedding,
        emotionalVector: emotionalVector,
        influencedWeights: influencedWeights
    };
}

// Generate a response based on thronglet state and vocabulary
function generateResponse(thronglet, integration) {
    if (!thronglet || !thronglet.agent) return "Beep?";
    
    // Get agent's current state
    const state = thronglet.agent.getState();
    
    // Calculate intelligence level (0-10) based on agent metrics
    let intelligenceLevel = 0;
    if (thronglet.agent.intelligenceScore !== undefined) {
        intelligenceLevel = Math.min(10, Math.floor(thronglet.agent.intelligenceScore * 2));
    }
    
    // Determine primary motivation based on state
    let primaryMotivation;
    if (thronglet.hunger > 70) {
        primaryMotivation = "hungry";
    } else if (thronglet.happiness < 40) {
        primaryMotivation = "unhappy";
    } else if (state.nearestThronglet < 0.3) {
        primaryMotivation = "social";
    } else {
        primaryMotivation = "curious";
    }
    
    // Calculate vocabulary access based on intelligence
    // Higher intelligence = access to more vocabulary
    const vocabLimit = 3 + Math.floor(intelligenceLevel * 1.5);
    
    // NEW: Check for recently taught/learned words to prioritize
    const recentWords = Array.from(vocabularyBase.values())
        .filter(word => word.learned && (Date.now() - word.learned < 3600000)) // Words learned in the last hour
        .sort((a, b) => b.learned - a.learned) // Most recently learned first
        .slice(0, 3); // Take up to 3 recent words
    
    // Get relevant vocab for the current motivation
    const relevantWords = Array.from(vocabularyBase.values()).filter(word => {
        // NEW: Include recently learned words regardless of context (50% chance)
        if (recentWords.includes(word) && Math.random() < 0.5) {
            return true;
        }
        
        if (primaryMotivation === "hungry" && 
            (word.contexts.includes("hungry") || word.contexts.includes("food") || 
             word.contexts.includes("apple") || word.contexts.includes("eat"))) {
            return true;
        }
        if (primaryMotivation === "unhappy" && 
            (word.contexts.includes("happy") || word.contexts.includes("play") || 
             word.contexts.includes("ball") || word.contexts.includes("fun"))) {
            return true;
        }
        if (primaryMotivation === "social" && 
            (word.contexts.includes("friend") || word.contexts.includes("together") || 
             word.contexts.includes("thronglet") || word.contexts.includes("hello"))) {
            return true;
        }
        if (primaryMotivation === "curious" && 
            (word.contexts.includes("explore") || word.contexts.includes("see") || 
             word.contexts.includes("what") || word.contexts.includes("where"))) {
            return true;
        }
        return false;
    });
    
    // Add value words (good/bad) if associated with the current motivation
    const valueWords = Array.from(vocabularyBase.values()).filter(word => {
        if (word.category === "value") {
            // Check if the value word is associated with the current motivation
            if (primaryMotivation === "hungry" && word.word === "good" && 
                thronglet.hunger < 30) {
                return true; // When not very hungry, eating is "good"
            }
            if (primaryMotivation === "hungry" && word.word === "bad" && 
                thronglet.hunger > 80) {
                return true; // When very hungry, it's "bad"
            }
            if (primaryMotivation === "unhappy" && word.word === "bad") {
                return true; // Being unhappy is "bad"
            }
            if (thronglet.happiness > 80 && word.word === "good") {
                return true; // Being very happy is "good"
            }
        }
        return false;
    });
    
    // NEW: Add random selection of words from general vocabulary (20% chance)
    let randomWords = [];
    if (Math.random() < 0.2) {
        randomWords = Array.from(vocabularyBase.values())
            .filter(word => !relevantWords.includes(word) && !valueWords.includes(word))
            .sort(() => 0.5 - Math.random()) // Shuffle
            .slice(0, 2); // Take up to 2 random words
    }
    
    // Sort by weight and limit by intelligence, but ensure recent words get priority
    let availableWords = [...relevantWords, ...valueWords, ...randomWords, ...recentWords]
        .sort((a, b) => {
            // Recent words get priority
            if (recentWords.includes(a) && !recentWords.includes(b)) return -1;
            if (!recentWords.includes(a) && recentWords.includes(b)) return 1;
            // Then sort by weight
            return b.weight - a.weight;
        })
        .slice(0, vocabLimit);
    
    // If not enough relevant words, add some general vocabulary
    if (availableWords.length < 3) {
        // ENHANCED: Improved general vocabulary selection to better include pronouns and verbs
        // First add essential communication words - prioritize pronouns and verbs
        const essentialWords = Array.from(vocabularyBase.values()).filter(word =>
            (word.category === "pronoun" || word.category === "verb") && 
            !availableWords.includes(word)
        ).sort((a, b) => b.weight - a.weight);
        
        // Add up to 2 essential communication words
        availableWords = [...availableWords, ...essentialWords.slice(0, 2)];
        
        // Then add other vocabulary if still needed
        if (availableWords.length < 3) {
            const otherWords = Array.from(vocabularyBase.values())
                .filter(word => 
                    !availableWords.includes(word) && 
                    word.category !== "pronoun" && 
                    word.category !== "verb"
                )
                .sort((a, b) => b.weight - a.weight)
                .slice(0, vocabLimit - availableWords.length);
            
            availableWords = [...availableWords, ...otherWords];
        }
    }
    
    // NEW: Add memory of last 3 responses to avoid repetition
    if (!thronglet.lastResponses) {
        thronglet.lastResponses = [];
    }
    
    // Generate multiple response options and choose the least repetitive one
    const responseOptions = [];
    
    // Generate 3 different response options
    for (let i = 0; i < 3; i++) {
        const responseOption = generateResponseVariation(
            thronglet, 
            intelligenceLevel, 
            primaryMotivation,
            availableWords,
            state
        );
        responseOptions.push(responseOption);
    }
    
    // Choose the response that's least similar to recent responses
    let bestResponse = responseOptions[0];
    let lowestSimilarity = 1;
    
    for (const option of responseOptions) {
        let maxSimilarity = 0;
        
        // Check similarity with recent responses
        for (const pastResponse of thronglet.lastResponses) {
            const similarity = calculateResponseSimilarity(option, pastResponse);
            maxSimilarity = Math.max(maxSimilarity, similarity);
        }
        
        // Pick the option with lowest similarity to past responses
        if (maxSimilarity < lowestSimilarity) {
            lowestSimilarity = maxSimilarity;
            bestResponse = option;
        }
    }
    
    // Update response history
    thronglet.lastResponses.unshift(bestResponse);
    if (thronglet.lastResponses.length > 3) {
        thronglet.lastResponses.pop();
    }
    
    // Update word weights based on usage
    const words = bestResponse.toLowerCase().split(/\s+/);
    words.forEach(word => {
        if (vocabularyBase.has(word)) {
            const vocabItem = vocabularyBase.get(word);
            vocabItem.weight = Math.min(1.0, vocabItem.weight + 0.02);
            vocabItem.lastUsed = Date.now();
        }
    });
    
    // We're modifying the last part that adds Morse code
    // Add Morse code for a key concept with higher intelligence
    if (intelligenceLevel >= 3 && Math.random() < 0.6) { // Increased probability
        // NEW: Try to find newly learned words to encode in Morse
        const responseWords = bestResponse.toLowerCase().split(/\s+/);
        
        // First look for recently learned words (in the last hour)
        let recentWords = responseWords.filter(w => 
            vocabularyBase.has(w) && 
            vocabularyBase.get(w).learned && 
            (Date.now() - vocabularyBase.get(w).learned < 3600000)
        );
        
        // If no recent words, look for any words with sufficient weight
        if (recentWords.length === 0) {
            recentWords = responseWords.filter(w => 
                vocabularyBase.has(w) && 
                vocabularyBase.get(w).weight >= 0.6 &&
                w.length >= 3
            );
        }
        
        if (recentWords.length > 0) {
            // Select a random key word with preference for newer words
            const keyWord = recentWords[Math.floor(Math.random() * recentWords.length)];
            
            // Get or generate Morse pattern
            let morsePattern = "";
            if (vocabularyBase.has(keyWord) && vocabularyBase.get(keyWord).morsePattern) {
                morsePattern = vocabularyBase.get(keyWord).morsePattern;
            } else {
                morsePattern = textToMorse(keyWord);
                
                // Save the pattern for future use
                if (vocabularyBase.has(keyWord)) {
                    vocabularyBase.get(keyWord).morsePattern = morsePattern;
                }
            }
            
            // Add the Morse code to the response
            if (morsePattern) {
                bestResponse += ` ${morsePattern}`;
                console.log(`Adding Morse code for "${keyWord}": ${morsePattern}`);
            }
        }
    }
    
    return bestResponse.trim();
}

// NEW: Function to calculate similarity between two responses
function calculateResponseSimilarity(responseA, responseB) {
    if (!responseA || !responseB) return 0;
    
    const wordsA = responseA.toLowerCase().split(/\s+/);
    const wordsB = responseB.toLowerCase().split(/\s+/);
    
    // Count common words
    let commonWords = 0;
    for (const word of wordsA) {
        if (wordsB.includes(word)) {
            commonWords++;
        }
    }
    
    // Calculate Jaccard similarity
    const uniqueWords = new Set([...wordsA, ...wordsB]);
    return commonWords / uniqueWords.size;
}

// NEW: Separate function to generate response variations
function generateResponseVariation(thronglet, intelligenceLevel, primaryMotivation, availableWords, state) {
    let response = "";
    
    // NEW: Add response templates for each intelligence level and motivation
    const templates = {
        // Very basic (just words, no grammar)
        basic: {
            hungry: ["food", "eat", "hungry", "apple want", "need food"],
            unhappy: ["play", "ball", "sad", "want play", "need happy"],
            social: ["friend", "hello", "together", "see you", "we friend"],
            curious: ["what", "see", "explore", "look there", "find new"]
        },
        // Simple phrases
        simple: {
            hungry: ["I want food", "I eat apple", "food good", "need eat now", "I hungry", "give food please"],
            unhappy: ["I need play", "I want ball", "play good", "ball make happy", "I sad", "need fun now"],
            social: ["I see friend", "we together", "hello you", "friend good", "we play", "you help me"],
            curious: ["I see what", "what over there", "I explore", "find new things", "what that", "I look"]
        },
        // Intermediate
        intermediate: {
            hungry: ["I am hungry now", "need food to eat", "want apple please", "hungry need food", "eating make good feeling"],
            unhappy: ["I not happy now", "want play with ball", "need fun things", "play make me happy", "sad feeling bad"],
            social: ["hello friend thronglet", "we can be together", "you and I friends", "we do things together", "friend help friend"],
            curious: ["I want explore world", "what is over there", "looking for new things", "want learn more", "what that thing is"]
        }
    };
    
    // Randomly select a suitable template based on intelligence and motivation
    if (intelligenceLevel <= 2) {
        // Very basic
        const template = templates.basic[primaryMotivation][Math.floor(Math.random() * templates.basic[primaryMotivation].length)];
        response = template;
    } 
    else if (intelligenceLevel <= 4) {
        // Simple phrases with pronouns and verbs
        // 70% chance to use a template, 30% to generate dynamically
        if (Math.random() < 0.7) {
            const template = templates.simple[primaryMotivation][Math.floor(Math.random() * templates.simple[primaryMotivation].length)];
            response = template;
            
            // Replace a word with random available word 40% of time for variety
            if (Math.random() < 0.4 && availableWords.length > 0) {
                const words = response.split(" ");
                const randomIndex = Math.floor(Math.random() * words.length);
                const randomWord = availableWords[Math.floor(Math.random() * availableWords.length)].word;
                words[randomIndex] = randomWord;
                response = words.join(" ");
            }
        } else {
            // Try to include a pronoun + verb + object structure
            const pronouns = availableWords.filter(w => w.category === "pronoun");
            const verbs = availableWords.filter(w => w.category === "verb");
            const values = availableWords.filter(w => w.category === "value");
            
            // Get pronoun (default to "I" if none available)
            let pronoun = "I";
            if (pronouns.length > 0) {
                // ENHANCEMENT: More varied pronoun selection - sometimes use we or my
                if (primaryMotivation === "social" && pronouns.some(p => p.word === "we")) {
                    pronoun = "we";
                } else if (Math.random() < 0.3 && pronouns.some(p => p.word === "my")) {
                    // Use possessive sometimes for variety
                    pronoun = "my";
                } else {
                    pronoun = pronouns[0].word;
                }
            }
            
            // Get verb (default based on motivation)
            let verb = primaryMotivation === "hungry" ? "want" : 
                     primaryMotivation === "unhappy" ? "need" :
                     primaryMotivation === "social" ? "see" : "explore";
            
            // ENHANCEMENT: Better verb selection based on context
            if (verbs.length > 0) {
                if (primaryMotivation === "hungry" && verbs.some(v => v.word === "eat")) {
                    verb = "eat";
                } else if (primaryMotivation === "unhappy" && verbs.some(v => v.word === "like")) {
                    verb = "like";
                } else if (primaryMotivation === "social" && verbs.some(v => v.word === "help")) {
                    verb = "help";
                } else if (Math.random() < 0.7) { // 70% chance to use a random verb
                    verb = verbs[Math.floor(Math.random() * verbs.length)].word;
                }
            }
            
            // Get object words
            const objects = availableWords.filter(w => 
                w.category !== "pronoun" && w.category !== "verb" && w.category !== "value"
            );
            
            // ENHANCEMENT: Handle possessive pronoun case
            if (pronoun === "my") {
                if (objects.length > 0) {
                    response = pronoun + " " + objects[0].word;
                    
                    // Add a verb after the object for better flow
                    if (verbs.length > 0) {
                        response += " " + verbs[0].word;
                    }
                } else {
                    // Fallback if no objects
                    response = "I " + verb;
                }
            } else {
                // Normal subject-verb-object structure
                response = pronoun + " " + verb;
                if (objects.length > 0) {
                    response += " " + objects[0].word;
                }
            }
            
            // Add value word if available (e.g., "I want food good")
            if (values.length > 0 && intelligenceLevel >= 3) {
                response += " " + values[0].word;
            }
        }
    }
    // Intermediate and Advanced levels
    else {
        // Use templates 50% of the time for variety
        if (Math.random() < 0.5) {
            const template = templates.intermediate[primaryMotivation][Math.floor(Math.random() * templates.intermediate[primaryMotivation].length)];
            response = template;
            
            // Add a detail or recently learned word
            if (availableWords.length > 0) {
                // Find recent or high-weight words
                const specialWords = availableWords.filter(w => 
                    w.learned && (Date.now() - w.learned < 3600000) || w.weight > 0.7
                );
                
                if (specialWords.length > 0) {
                    const specialWord = specialWords[Math.floor(Math.random() * specialWords.length)].word;
                    
                    // Different ways to incorporate the special word
                    const incorporations = [
                        ` with ${specialWord}`,
                        `. ${specialWord} is important`,
                        `. I like ${specialWord}`,
                        ` and ${specialWord}`,
                        ` because ${specialWord}`
                    ];
                    
                    response += incorporations[Math.floor(Math.random() * incorporations.length)];
                }
            }
        } else {
            // Default to original complex sentence generation
            if (primaryMotivation === "hungry") {
                response = "I am hungry. ";
                
                // Add qualifier about food
                if (availableWords.some(w => w.word === "apple")) {
                    response += "Want apple";
                    // Add value association if learned
                    if (availableWords.some(w => w.word === "good")) {
                        response += " good";
                    }
                    response += ".";
                } else {
                    response += "Need food";
                    // Add value association if learned
                    if (availableWords.some(w => w.word === "good")) {
                        response += " good";
                    }
                    response += ".";
                }
            }
            else if (primaryMotivation === "unhappy") {
                response = "I not happy. ";
                
                // Add qualifier about play
                if (availableWords.some(w => w.word === "ball")) {
                    response += "Want play with ball";
                    // Add value association if learned
                    if (availableWords.some(w => w.word === "good")) {
                        response += " good";
                    }
                    response += ".";
                } else {
                    response += "Need fun";
                    // Add value association if learned
                    if (availableWords.some(w => w.word === "good")) {
                        response += " good";
                    }
                    response += ".";
                }
                
                // Add negative value if unhappy is really bad
                if (thronglet.happiness < 20 && availableWords.some(w => w.word === "bad")) {
                    response += " Unhappy bad.";
                }
            }
            else if (primaryMotivation === "social") {
                if (availableWords.some(w => w.word === "hello")) {
                    response = "Hello friend. ";
                } else {
                    response = "I see you. ";
                }
                
                // Add social qualifier
                if (availableWords.some(w => w.word === "together")) {
                    response += "We together";
                    // Add value if sociality is good
                    if (availableWords.some(w => w.word === "good")) {
                        response += " good";
                    }
                    response += ".";
                }
            }
            else {
                response = "I want explore. ";
                
                // Add curious qualifier
                if (availableWords.some(w => w.word === "what")) {
                    response += "What over there?";
                }
            }
        }
    }
    
    return response;
}

// Reset language system vocabulary
function resetLanguageSystem() {
    vocabularyBase.clear();
    languageMemory = [];
    initializeVocabulary();
    console.log("Language system reset to initial state");
}

// Teach a specific word to the Thronglet with provided associations
function teachWord(word, associations = {}) {
    if (!languageEnabled || !word) return false;
    
    word = word.toLowerCase().trim();
    if (word.length === 0) return false;
    
    // Track if this is a new word
    let isNewWord = !wordAssociations[word];
    
    // Initialize word if not exists
    if (!wordAssociations[word]) {
        wordAssociations[word] = {
            weight: 0.5,
            associations: {},
            morseVersion: textToMorse(word) // Store Morse code version of the word
        };
    }
    
    // Increase word weight each time it's taught
    wordAssociations[word].weight = Math.min(1, wordAssociations[word].weight + 0.05);
    
    // Add or strengthen associations
    for (const [assoc, strength] of Object.entries(associations)) {
        if (!wordAssociations[word].associations[assoc]) {
            wordAssociations[word].associations[assoc] = 0;
        }
        
        // Strengthen the association
        wordAssociations[word].associations[assoc] = 
            Math.min(1, wordAssociations[word].associations[assoc] + strength);
    }
    
    // Check if the word is given in Morse code
    const morseVersion = textToMorse(word);
    
    // Add automatic association with Morse code
    if (!wordAssociations[word].associations["morse"]) {
        wordAssociations[word].associations["morse"] = 0.4;
    }
    
    // Also teach the Morse code version of this word
    if (!wordAssociations[morseVersion]) {
        wordAssociations[morseVersion] = {
            weight: 0.3,
            associations: {
                [word]: 0.9,  // Strong association to the original word
                "morse": 0.9  // Strong association to the concept of Morse
            },
            isMoreCode: true
        };
    } else {
        wordAssociations[morseVersion].associations[word] = 0.9;
        wordAssociations[morseVersion].associations["morse"] = 0.9;
    }
    
    // Add to language memory
    updateLanguageMemory(word, associations);
    
    // IMPORTANT FIX: Ensure the word is also added to vocabularyBase
    // This ensures taught words appear in vocabulary stats
    let vocabIsNew = false;
    if (!vocabularyBase.has(word)) {
        vocabIsNew = true;
        // Add to vocabularyBase
        vocabularyBase.set(word, {
            word: word,
            category: "taught",
            contexts: ["teaching"],
            weight: 0.5,
            learnedFrom: "direct_teaching",
            learned: Date.now(),
            lastUsed: Date.now(),
            associations: [],
            morsePattern: morseVersion
        });
        console.log(`Added "${word}" to vocabularyBase for stats tracking`);
    } else {
        // Update existing entry
        const vocabEntry = vocabularyBase.get(word);
        vocabEntry.weight = Math.min(1.0, vocabEntry.weight + 0.05);
        vocabEntry.lastUsed = Date.now();
        
        // Add "teaching" context if not present
        if (!vocabEntry.contexts.includes("teaching")) {
            vocabEntry.contexts.push("teaching");
        }
    }
    
    // Find embeddings for this word when the model is loaded
    if (languageModel && modelLoaded) {
        findWordEmbeddings([word, morseVersion]);
    }
    
    // Activate the word in Thronglets if it's new or directly taught
    if (window.thronglets && (isNewWord || vocabIsNew)) {
        // Pick a random Thronglet to vocalize the word
        const randomIndex = Math.floor(Math.random() * window.thronglets.length);
        const randomThronglet = window.thronglets[randomIndex];
        
        // Have the chosen Thronglet vocalize the word
        setTimeout(() => {
            activateWordInBrain(word, randomThronglet.id);
        }, 500); // Short delay
    }
    
    return true;
}

// Get vocabulary statistics
function getVocabularyStats() {
    const stats = {
        totalWords: vocabularyBase.size,
        wordsByCategory: {},
        mostUsedWords: [],
        recentlyLearnedWords: []
    };
    
    // Count words by category
    for (const [_, word] of vocabularyBase) {
        if (!stats.wordsByCategory[word.category]) {
            stats.wordsByCategory[word.category] = 0;
        }
        stats.wordsByCategory[word.category]++;
    }
    
    // Get most used words (by weight)
    stats.mostUsedWords = Array.from(vocabularyBase.values())
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 10)
        .map(w => ({
            word: w.word,
            weight: w.weight,
            category: w.category
        }));
    
    // Get recently learned words
    stats.recentlyLearnedWords = Array.from(vocabularyBase.values())
        .filter(w => w.learned)
        .sort((a, b) => b.learned - a.learned)
        .slice(0, 10)
        .map(w => ({
            word: w.word,
            learnedFrom: w.learnedFrom,
            learnedAt: new Date(w.learned).toLocaleString()
        }));
    
    return stats;
}

// Get full vocabulary list
function getFullVocabulary() {
    return Array.from(vocabularyBase.values());
}

// Modify the recordLanguageEvent function to have more diverse learning events
function recordLanguageEvent(eventType, throngletId) {
    if (!eventType || !throngletId) return;
    
    // Process specific event types with more varied language
    switch(eventType) {
        case "apple_eaten":
            // Learn eating-related vocabulary through experience
            learnWord("eat", "action", throngletId);
            learnWord("apple", "food", throngletId);
            learnWord("hungry", "state", throngletId);
            
            // Learn through contextual sentences instead of hard-coded associations
            learnFromSentence("I eat apple", "experience", throngletId);
            learnFromSentence("eat food good", "experience", throngletId);
            learnFromSentence("apple is food", "experience", throngletId);
            
            // 30% chance to learn a new food-related word
            if (Math.random() < 0.3) {
                const foodWords = ["tasty", "yummy", "sweet", "delicious", "juice", "fruit"];
                const randomWord = foodWords[Math.floor(Math.random() * foodWords.length)];
                learnWord(randomWord, "food", throngletId);
                learnFromSentence(`apple ${randomWord} good`, "experience", throngletId);
            }
            break;
            
        case "ball_played":
            // Learn play-related vocabulary through experience
            learnWord("ball", "toy", throngletId);
            learnWord("fun", "feeling", throngletId);
            learnWord("happy", "state", throngletId);
            
            // Learn through contextual sentences
            learnFromSentence("I play with ball", "experience", throngletId);
            learnFromSentence("playing is fun", "experience", throngletId);
            learnFromSentence("ball fun good", "experience", throngletId);
            
            // 30% chance to learn a new play-related word
            if (Math.random() < 0.3) {
                const playWords = ["game", "catch", "throw", "bounce", "roll", "kick"];
                const randomWord = playWords[Math.floor(Math.random() * playWords.length)];
                learnWord(randomWord, "play", throngletId);
                learnFromSentence(`ball ${randomWord} fun`, "experience", throngletId);
            }
            break;
            
        case "very_hungry":
            // Learn hunger-related vocabulary through experience
            learnWord("hungry", "feeling", throngletId);
            learnWord("need", "verb", throngletId);
            
            // Learn through contextual sentences
            learnFromSentence("I feel hungry", "experience", throngletId);
            learnFromSentence("need food", "experience", throngletId);
            learnFromSentence("hungry bad", "experience", throngletId);
            break;
            
        case "very_happy":
            // Learn happiness-related vocabulary through experience
            learnWord("happy", "feeling", throngletId);
            learnWord("feel", "verb", throngletId);
            
            // Learn through contextual sentences
            learnFromSentence("I feel happy", "experience", throngletId);
            learnFromSentence("play makes happy", "experience", throngletId);
            learnFromSentence("happy good", "experience", throngletId);
            break;
            
        case "thronglet_nearby":
            // Learn social vocabulary through experience
            learnWord("friend", "social", throngletId);
            learnWord("we", "pronoun", throngletId);
            
            // Learn through contextual sentences
            learnFromSentence("you are friend", "experience", throngletId);
            learnFromSentence("we together", "experience", throngletId);
            
            // 20% chance to learn a new social word
            if (Math.random() < 0.2) {
                const socialWords = ["hello", "hi", "greet", "meet", "together", "help"];
                const randomWord = socialWords[Math.floor(Math.random() * socialWords.length)];
                learnWord(randomWord, "social", throngletId);
                learnFromSentence(`${randomWord} friend good`, "experience", throngletId);
            }
            break;
    }
}

// Morse code integration with language learning system
function processMorseCode(morsePattern, throngletId) {
    if (!morsePattern || !throngletId) return null;
    
    // Check if system is enabled and loaded
    if (!languageEnabled || !modelLoaded) return null;
    
    // Convert Morse code to text
    const text = morseToText(morsePattern);
    if (!text) return null;
    
    // Learn this word/phrase with special Morse context
    learnFromSentence(text, "morse_communication", throngletId);
    
    // Create a stronger association between this word and its Morse pattern
    if (vocabularyBase.has(text)) {
        const wordObj = vocabularyBase.get(text);
        if (!wordObj.morsePattern) {
            wordObj.morsePattern = morsePattern;
        }
        
        // Increase the weight for Morse-learned words
        wordObj.weight = Math.min(1.0, wordObj.weight + 0.1);
        console.log(`Reinforced word "${text}" through Morse code pattern: ${morsePattern}`);
    }
    
    return text;
}

// Convert text to Morse code for language outputs
function getConceptAsMorse(concept) {
    if (!concept) return null;
    
    // Convert to lowercase for consistency
    concept = concept.toLowerCase().trim();
    
    // Check if this concept already has a stored Morse pattern
    if (vocabularyBase.has(concept)) {
        const wordObj = vocabularyBase.get(concept);
        if (wordObj.morsePattern) {
            return wordObj.morsePattern;
        }
    }
    
    // Generate Morse code for this concept
    const morsePattern = textToMorse(concept);
    
    // Save the pattern for future use
    if (vocabularyBase.has(concept)) {
        vocabularyBase.get(concept).morsePattern = morsePattern;
        console.log(`Generated and saved Morse pattern for "${concept}": ${morsePattern}`);
    }
    
    return morsePattern;
}

// Generate a language response for a given thronglet
async function generateThrongletLanguageResponse(thronglet, input) {
    if (!input || input.trim() === '') return '';
    
    input = input.toLowerCase().trim();
    let response = '';
    
    // Extract any potential Morse code from the input
    const morsePatterns = extractMorsePatterns(input);
    let decodedWords = [];
    
    // Decode any Morse code and teach the thronglet these words
    for (const pattern of morsePatterns) {
        const decoded = morseToText(pattern);
        if (decoded && decoded.length > 0) {
            decodedWords.push(decoded);
            
            // Teach the thronglet the decoded word and associate it with the Morse pattern
            teachWord(thronglet, decoded);
            
            // Create or strengthen association with Morse code
            const morseAssociation = pattern.replace(/\s+/g, '_');
            updateAssociationStrength(decoded, 'morse_' + morseAssociation, 0.9);
        }
    }
    
    // First, check for known words in the input
    const knownWords = extractKnownWords(input);
    const knownWordsResponse = handleKnownWords(thronglet, knownWords);
    
    // Next, use embeddings for understanding the semantic meaning
    const embedding = await getEmbedding(input);
    const similarWords = await findSimilarWords(embedding);
    const embeddingResponse = handleEmbeddingResponse(thronglet, similarWords);
    
    // Combine responses with a bit of randomness
    if (knownWords.length > 0 && Math.random() < 0.7) {
        response = knownWordsResponse;
    } else if (similarWords.length > 0) {
        response = embeddingResponse;
    } else {
        // Fallback to a default response
        const defaultResponses = [
            "I'm trying to understand...",
            "Interesting...",
            "Tell me more.",
            "I'm learning."
        ];
        response = selectRandomPhrase(defaultResponses);
    }
    
    // Occasionally convert some known words in the response to Morse code
    if (Math.random() < 0.3) {
        response = convertResponseToPartialMorse(response);
    }
    
    // Record the interaction in language memory
    thronglet.languageMemory.push({
        input: input,
        response: response,
        timestamp: Date.now()
    });
    
    // Trim memory if it gets too large
    if (thronglet.languageMemory.length > 20) {
        thronglet.languageMemory.shift();
    }
    
    return response;
}

// Handle known words in the input
function handleKnownWords(thronglet, knownWords) {
    if (knownWords.length === 0) return '';
    
    // Prioritize responding to action words
    const actionWord = knownWords.find(word => 
        word === 'eat' || word === 'play' || word === 'sleep');
    
    if (actionWord) {
        thronglet.performAction(actionWord);
        
        const responses = {
            'eat': ["Food! Good!", "Yum! Eat!", "I like food!", ".-. -- .-- ..."],  // Morse for "yum"
            'play': ["Play fun!", "Fun! Play!", "I like play!", "...- ..- -."],    // Morse for "fun"
            'sleep': ["Sleep good.", "Tired. Sleep.", "Need rest.", "-.. .-. . .-. --"]  // Morse for "dream"
        };
        
        return selectRandomPhrase(responses[actionWord]);
    }
    
    // Respond to value words
    const valueWord = knownWords.find(word => 
        word === 'good' || word === 'bad');
    
    if (valueWord) {
        const responses = {
            'good': ["Yes, good!", "Good! Happy!", "I like good!", "--. --- --- -.."],  // Morse for "good"
            'bad': ["No bad.", "Bad! Sad!", "I no like bad.", "-... .- -.."]   // Morse for "bad"
        };
        
        return selectRandomPhrase(responses[valueWord]);
    }
    
    // General response for other known words
    const responseWords = [];
    for (const word of knownWords) {
        const associations = getTopAssociations(word, 2);
        responseWords.push(...associations.map(a => a.target));
    }
    
    if (responseWords.length > 0) {
        // Construct a simple response using the associated words
        return responseWords.slice(0, 3).join(' ') + '!';
    }
    
    return knownWords[0] + '...';
}

// Handle response based on embedding similarity
function handleEmbeddingResponse(thronglet, similarWords) {
    if (similarWords.length === 0) return '';
    
    const responses = [];
    
    for (const word of similarWords) {
        const associations = getTopAssociations(word, 2);
        
        if (associations.length > 0) {
            const phrase = word + ' ' + associations[0].target;
            responses.push(phrase);
        } else {
            responses.push(word);
        }
    }
    
    return responses.slice(0, 2).join(', ') + '?';
}

// Convert some words in response to Morse code based on their associations
function convertResponseToPartialMorse(response) {
    const words = response.split(/\s+/);
    
    // NEW: Ensure we get at least 1 word converted but not more than half the sentence
    const maxMorseWords = Math.max(1, Math.floor(words.length / 2));
    const morseCount = Math.floor(Math.random() * maxMorseWords) + 1;
    const morseIndices = [];
    
    // NEW: Track which words have Morse patterns in vocabularyBase
    const wordsWithMorse = [];
    
    // First, check which words have proper Morse patterns in the vocabulary
    words.forEach((word, index) => {
        if (vocabularyBase.has(word.toLowerCase())) {
            const wordObj = vocabularyBase.get(word.toLowerCase());
            if (wordObj.morsePattern) {
                wordsWithMorse.push(index);
            }
        }
    });
    
    // NEW: Prioritize newly learned words for Morse conversion
    if (wordsWithMorse.length > 0) {
        // Use at least one word with known Morse pattern
        morseIndices.push(wordsWithMorse[Math.floor(Math.random() * wordsWithMorse.length)]);
        
        // Add more random words if needed
        while (morseIndices.length < morseCount && morseIndices.length < words.length) {
            const index = Math.floor(Math.random() * words.length);
            if (!morseIndices.includes(index)) {
                morseIndices.push(index);
            }
        }
    } else {
        // If no words have Morse patterns, just pick random indices
        while (morseIndices.length < morseCount) {
            const index = Math.floor(Math.random() * words.length);
            if (!morseIndices.includes(index)) {
                morseIndices.push(index);
            }
        }
    }
    
    // Convert selected words to Morse
    for (const index of morseIndices) {
        const word = words[index].toLowerCase();
        
        // NEW: Prioritize words in vocabulary for more accurate Morse patterns
        if (vocabularyBase.has(word)) {
            const vocabWord = vocabularyBase.get(word);
            if (vocabWord.morsePattern) {
                words[index] = vocabWord.morsePattern;
                console.log(`Using vocabulary Morse pattern for "${word}": ${vocabWord.morsePattern}`);
                continue;
            }
        }
        
        // Fallback: check if the word has a Morse association in wordAssociations
        if (wordAssociations[word] && wordAssociations[word].associations) {
            const associations = Object.keys(wordAssociations[word].associations)
                .filter(assoc => assoc.startsWith('morse_'));
            
            if (associations.length > 0) {
                // Use the existing Morse association
                const morseCode = associations[0].substring(6).replace(/_/g, ' ');
                words[index] = morseCode;
                console.log(`Using association Morse pattern for "${word}": ${morseCode}`);
                continue;
            }
        }
        
        // Last resort: Generate new Morse code
        const morseCode = textToMorse(word);
        words[index] = morseCode;
        
        // NEW: Save the Morse pattern to the vocabulary for future use
        if (vocabularyBase.has(word)) {
            vocabularyBase.get(word).morsePattern = morseCode;
            console.log(`Generated and saved new Morse pattern for "${word}": ${morseCode}`);
        }
    }
    
    return words.join(' ');
}

// Update the strength of an association between a word and a concept
function updateAssociationStrength(word, associatedConcept, strength) {
    if (!word || !associatedConcept) return false;
    
    // Check if the word exists in our vocabulary
    if (wordAssociations[word]) {
        // Initialize associations object if it doesn't exist
        if (!wordAssociations[word].associations) {
            wordAssociations[word].associations = {};
        }
        
        // Update or create the association
        if (wordAssociations[word].associations[associatedConcept]) {
            // Strengthen existing association
            wordAssociations[word].associations[associatedConcept] = 
                Math.min(1.0, wordAssociations[word].associations[associatedConcept] + 0.1);
        } else {
            // Create new association
            wordAssociations[word].associations[associatedConcept] = strength;
        }
        
        console.log(`Updated association: "${word}" -> "${associatedConcept}" (${wordAssociations[word].associations[associatedConcept]})`);
        return true;
    }
    
    return false;
}

// Get top associations for a word
function getTopAssociations(word, count = 2) {
    if (!wordAssociations[word] || !wordAssociations[word].associations) {
        return [];
    }
    
    // Get all associations
    const associations = Object.entries(wordAssociations[word].associations)
        .map(([target, strength]) => ({ target, strength }))
        .filter(assoc => !assoc.target.startsWith('morse_')) // Filter out morse associations
        .sort((a, b) => b.strength - a.strength)
        .slice(0, count);
    
    return associations;
}

// Update language memory with new words and associations
function updateLanguageMemory(word, associations = {}) {
    if (!word) return;
    
    // Add the word to language memory
    languageMemory.push({
        word: word,
        associations: associations,
        timestamp: Date.now()
    });
    
    // Limit memory size
    if (languageMemory.length > 100) {
        languageMemory.shift();
    }
    
    // Log for debugging
    console.log(`Added word "${word}" to language memory with ${Object.keys(associations).length} associations`);
}

// Find word embeddings
async function findWordEmbeddings(words) {
    if (!languageModel || !modelLoaded) return;
    
    try {
        for (const word of words) {
            if (!word || typeof word !== 'string' || word.trim() === '') continue;
            
            // Skip if already have embedding
            if (wordEmbeddings[word]) continue;
            
            // Generate embedding for this word
            const embedding = await getEmbedding(word);
            if (embedding) {
                wordEmbeddings[word] = embedding;
                console.log(`Generated embedding for word: ${word}`);
            }
        }
    } catch (error) {
        console.error("Error generating word embeddings:", error);
    }
}

/**
 * Enable or disable the auto-teaching system
 * @param {boolean} enabled - Whether to enable auto-teaching
 */
function setAutoTeachEnabled(enabled) {
    isAutoTeachEnabled = enabled;
    
    if (enabled && !autoTeachInterval) {
        // Start auto-teaching with rapid initial frequency
        autoTeachInterval = setInterval(autoTeachWords, AUTO_TEACH_INITIAL_FREQUENCY);
        autoTeachStartTime = Date.now();
        console.log("Auto-teaching system enabled with rapid learning");
        
        // Do an immediate teaching session
        autoTeachWords();
        
        // Set up a checker to adjust the frequency after the initial period
        setTimeout(() => {
            if (isAutoTeachEnabled && autoTeachInterval) {
                // Clear the rapid interval
                clearInterval(autoTeachInterval);
                
                // Set up the slower interval
                autoTeachInterval = setInterval(autoTeachWords, AUTO_TEACH_LATER_FREQUENCY);
                console.log("Auto-teaching system slowed down after initial learning period");
            }
        }, AUTO_TEACH_INITIAL_PERIOD);
        
    } else if (!enabled && autoTeachInterval) {
        // Stop auto-teaching
        clearInterval(autoTeachInterval);
        autoTeachInterval = null;
        autoTeachStartTime = null;
        console.log("Auto-teaching system disabled");
    }
}

/**
 * Get the current vocabulary as a formatted string for the LLM
 * @returns {string} - Formatted vocabulary information
 */
function getFormattedVocabularyForLLM() {
    let result = "CURRENT VOCABULARY:\n";
    
    // Add known words with their associations
    vocabularyBase.forEach((wordData, word) => {
        result += `- "${word}": `;
        
        const associations = [];
        if (wordData.associations && Array.isArray(wordData.associations)) {
            wordData.associations.forEach(assoc => {
                associations.push(`${assoc.word} (strength: ${assoc.strength.toFixed(2)})`);
            });
        }
        
        if (associations.length > 0) {
            result += associations.join(", ");
        } else {
            result += "no strong associations yet";
        }
        result += "\n";
    });
    
    // Add information about known concepts
    result += "\nKNOWN CONCEPTS:\n";
    ["food", "happy", "sad", "follow"].forEach(concept => {
        if (vocabularyBase.has(concept)) {
            result += `- "${concept}"\n`;
        }
    });
    
    return result;
}

/**
 * Get information about Thronglet states for context
 * @returns {string} - Formatted Thronglet state information
 */
function getThrongletStatesInfo() {
    let result = "THRONGLET STATES:\n";
    
    // Access the global thronglets array from the window
    if (window.thronglets && window.thronglets.length > 0) {
        window.thronglets.forEach(thronglet => {
            result += `- Thronglet #${thronglet.id}: hunger=${thronglet.hunger}, happiness=${thronglet.happiness}\n`;
        });
    } else {
        result += "No Thronglets currently active\n";
    }
    
    return result;
}

/**
 * Truncate context to prevent exceeding token limits
 * @param {string} text - Text to truncate
 * @returns {string} - Truncated text
 */
function truncateContext(text) {
    const estimatedTokens = text.length / 4; // Rough estimate: 4 chars per token
    
    if (estimatedTokens > MAX_CONTEXT_LENGTH) {
        console.log(`Context too long (est. ${estimatedTokens} tokens), truncating...`);
        // Keep only the first part of the context
        const truncatedText = text.substring(0, MAX_CONTEXT_LENGTH * 4);
        return truncatedText + "\n[Context truncated due to length]";
    }
    
    return text;
}

/**
 * Call the Fireworks API to get suggestions for new words
 * @returns {Promise<Array>} - Array of suggested words and associations
 */
async function getWordSuggestionsFromLLM() {
    if (!isAutoTeachEnabled) return [];
    
    try {
        // Prepare the context
        const vocabContext = getFormattedVocabularyForLLM();
        const throngletContext = getThrongletStatesInfo();
        const teachingHistoryContext = getRecentTeachingHistory();
        
        const prompt = `
You are an AI assistant helping Thronglets (cute virtual creatures) learn language. 
Your task is to suggest new words to teach them based on their current vocabulary and state.

${vocabContext}

${throngletContext}

${teachingHistoryContext}

INSTRUCTIONS:
1. Suggest 1-3 new words that would be valuable additions to the Thronglets' vocabulary
2. DO NOT suggest any words that were recently taught (listed in the teaching history)
3. Each suggested word should be UNIQUE and not already in the vocabulary
4. Each new word should logically build on the existing vocabulary
5. For each word, provide 2-4 associated concepts or words from the existing vocabulary
6. IMPORTANT: Include either "good" or "bad" as one of the associations to help Thronglets form value judgments
7. Explain briefly why each word would be useful for the Thronglets
8. Format your response as a JSON array with this structure:
[
  {
    "word": "example",
    "associations": {
      "related_word1": 0.8,
      "related_word2": 0.6,
      "good": 0.7
    },
    "reason": "Brief explanation of why this word is useful"
  }
]
9. The strength of association should be between 0.1 and 0.9

Please respond ONLY with the JSON array and nothing else.`;

        // Truncate context if needed
        const truncatedPrompt = truncateContext(prompt);
        
        // Call Fireworks API
        const response = await fetch(FIREWORKS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${FIREWORKS_API_KEY}`
            },
            body: JSON.stringify({
                model: FIREWORKS_MODEL,
                messages: [
                    {
                        role: "user",
                        content: truncatedPrompt
                    }
                ],
                temperature: 0.7,
                max_tokens: 1000
            })
        });
        
        if (!response.ok) {
            throw new Error(`API response error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Extract the JSON response
        const content = data.choices[0].message.content;
        console.log("Raw LLM response:", content);
        
        // More robust JSON extraction - try multiple approaches
        let suggestions = [];
        
        try {
            // Try to directly parse the entire content
            suggestions = JSON.parse(content);
        } catch (parseError) {
            console.log("Direct JSON parsing failed, trying extraction methods");
            
            try {
                // Try to extract JSON using regex (look for array pattern)
                const jsonMatch = content.match(/\[\s*\{[\s\S]*\}\s*\]/);
                if (jsonMatch) {
                    suggestions = JSON.parse(jsonMatch[0]);
                } else {
                    // Try to find individual objects and combine them
                    const objectMatches = content.match(/\{\s*"word"[\s\S]*?\}\s*(?=,|\]|$)/g);
                    if (objectMatches && objectMatches.length > 0) {
                        // Combine objects into a proper array
                        const jsonStr = '[' + objectMatches.join(',') + ']';
                        suggestions = JSON.parse(jsonStr);
                    } else {
                        // Last resort - try to manually extract the word suggestions
                        // Look for patterns like "word": "example"
                        const wordMatches = content.match(/"word"\s*:\s*"([^"]*)"/g);
                        if (wordMatches && wordMatches.length > 0) {
                            // Create basic suggestion objects with just the words
                            suggestions = wordMatches.map(match => {
                                const word = match.match(/"word"\s*:\s*"([^"]*)"/)[1];
                                return { 
                                    word: word, 
                                    associations: { "good": 0.7 },
                                    reason: "Auto-extracted from malformed JSON"
                                };
                            });
                        }
                    }
                }
            } catch (extractError) {
                console.error("All JSON extraction methods failed:", extractError);
                // Create a fallback suggestion if everything fails
                suggestions = [
                    {
                        word: "fallback",
                        associations: {
                            "good": 0.7,
                            "try": 0.6,
                            "again": 0.5
                        },
                        reason: "Created as fallback due to JSON parsing errors"
                    }
                ];
            }
        }
        
        // Validate suggestions format
        if (!Array.isArray(suggestions)) {
            console.warn("Suggestions is not an array, converting");
            if (typeof suggestions === 'object') {
                // If a single object was returned, convert to array
                suggestions = [suggestions];
            } else {
                suggestions = [];
            }
        }
        
        // Filter out invalid suggestions
        suggestions = suggestions.filter(suggestion => 
            suggestion && typeof suggestion === 'object' && 
            suggestion.word && typeof suggestion.word === 'string');
        
        // Ensure each suggestion has the required fields
        suggestions = suggestions.map(suggestion => {
            if (!suggestion.associations || typeof suggestion.associations !== 'object') {
                suggestion.associations = { "good": 0.7 };
            }
            if (!suggestion.reason) {
                suggestion.reason = "Automatically suggested word";
            }
            return suggestion;
        });
        
        console.log("Processed word suggestions:", suggestions);
        return suggestions;
    } catch (error) {
        console.error("Error getting word suggestions from LLM:", error);
        // Return a default suggestion as fallback
        return [{
            word: "retry",
            associations: {
                "try": 0.8,
                "again": 0.7,
                "good": 0.6
            },
            reason: "Created as emergency fallback due to API error"
        }];
    }
}

/**
 * Get formatted string of recently taught words for LLM context
 * @returns {string} Formatted teaching history
 */
function getRecentTeachingHistory() {
    let result = "RECENTLY TAUGHT WORDS (do not repeat these):\n";
    
    if (autoTaughtWords.length === 0) {
        result += "No words have been auto-taught yet.\n";
    } else {
        // Format the auto-taught words with timestamps
        autoTaughtWords.forEach((item, index) => {
            const timeAgo = Math.floor((Date.now() - item.timestamp) / 60000); // Minutes ago
            result += `${index + 1}. "${item.word}" (taught ${timeAgo} minutes ago)\n`;
        });
    }
    
    return result;
}

/**
 * Automatically teach new words to Thronglets using the Fireworks LLM
 */
async function autoTeachWords() {
    if (!isAutoTeachEnabled || !modelLoaded) return;
    
    console.log("Auto-teaching system running...");
    
    try {
        // Occasionally practice example phrases instead of teaching new words
        // This helps Thronglets learn coherent sentence structures
        if (Math.random() < 0.3 && vocabularyBase.size > 10) { // 30% chance if vocabulary is large enough
            console.log("Getting example phrases for practice instead of new words");
            
            // First, collect state information about Thronglets to make contextual phrases
            let contextualStates = [];
            let selectedThronglet = null;
            
            if (window.thronglets && window.thronglets.length > 0) {
                // Count how many Thronglets are in various states
                let hungryCount = 0;
                let veryHungryCount = 0;
                let happyCount = 0;
                let sadCount = 0;
                
                window.thronglets.forEach(thronglet => {
                    if (thronglet.hunger > 70) veryHungryCount++;
                    else if (thronglet.hunger > 50) hungryCount++;
                    
                    if (thronglet.happiness < 30) sadCount++;
                    else if (thronglet.happiness > 70) happyCount++;
                });
                
                // Calculate percentages based on total Thronglet count
                const totalThronglets = window.thronglets.length;
                if (veryHungryCount > 0) {
                    contextualStates.push({
                        state: "very hungry",
                        value: Math.round((veryHungryCount / totalThronglets) * 100)
                    });
                }
                
                if (hungryCount > 0) {
                    contextualStates.push({
                        state: "hungry",
                        value: Math.round((hungryCount / totalThronglets) * 100)
                    });
                }
                
                if (sadCount > 0) {
                    contextualStates.push({
                        state: "sad",
                        value: Math.round((sadCount / totalThronglets) * 100)
                    });
                }
                
                if (happyCount > 0) {
                    contextualStates.push({
                        state: "happy",
                        value: Math.round((happyCount / totalThronglets) * 100)
                    });
                }
                
                // Choose a random Thronglet to focus on (prioritize those with stronger needs)
                // First look for very hungry or very sad Thronglets
                const needyThronglets = window.thronglets.filter(t => 
                    t.hunger > 70 || t.happiness < 30
                );
                
                if (needyThronglets.length > 0) {
                    // Prioritize needier Thronglets for phrase practice
                    selectedThronglet = needyThronglets[Math.floor(Math.random() * needyThronglets.length)];
                } else {
                    // Otherwise pick a random Thronglet
                    selectedThronglet = window.thronglets[Math.floor(Math.random() * window.thronglets.length)];
                }
            }
            
            // Build context object for phrase generation
            const phraseContext = {
                states: contextualStates,
                selectedThronglet: selectedThronglet
            };
            
            // Get contextually relevant example phrases
            const examplePhrases = await getExamplePhrasesFromLLM(phraseContext);
            
            // If we got valid phrases and have Thronglets, practice them
            if (examplePhrases.length > 0 && window.thronglets && window.thronglets.length > 0) {
                console.log(`Received ${examplePhrases.length} example phrases for practice`);
                
                // Prioritize phrases relevant to the selected Thronglet's state
                let prioritizedPhrases = [...examplePhrases];
                
                if (selectedThronglet) {
                    // Put phrases matching the Thronglet's needs first
                    if (selectedThronglet.hunger > 70) {
                        prioritizedPhrases.sort((a, b) => 
                            (b.context === "hunger" ? 1 : 0) - (a.context === "hunger" ? 1 : 0)
                        );
                    } else if (selectedThronglet.happiness < 30) {
                        prioritizedPhrases.sort((a, b) => 
                            (b.context === "happiness" ? 1 : 0) - (a.context === "happiness" ? 1 : 0)
                        );
                    }
                }
                
                // Pick 1-3 phrases to practice, with priority given to contextual ones
                const phrasesToPractice = prioritizedPhrases.slice(0, Math.min(3, prioritizedPhrases.length));
                
                // Have the selected Thronglet practice the most relevant phrase first
                if (selectedThronglet && phrasesToPractice.length > 0) {
                    // Practice the first phrase immediately with the selected Thronglet
                    practicePhrase(phrasesToPractice[0].phrase, selectedThronglet.id);
                    
                    // Have other Thronglets practice the remaining phrases
                    for (let i = 1; i < phrasesToPractice.length; i++) {
                        setTimeout(() => {
                            // Find Thronglets that match this phrase's context
                            const matchingThronglets = window.thronglets.filter(t => {
                                if (phrasesToPractice[i].context === "hunger") {
                                    return t.hunger > 50; // Hungry Thronglets
                                } else if (phrasesToPractice[i].context === "happiness") {
                                    return t.happiness < 50 || t.happiness > 70; // Sad or happy Thronglets
                                } else {
                                    return true; // Any Thronglet for general phrases
                                }
                            });
                            
                            // Choose a random matching Thronglet if available, otherwise any Thronglet
                            const throngletsToPick = matchingThronglets.length > 0 ? 
                                matchingThronglets : window.thronglets;
                            
                            // Exclude the already speaking Thronglet
                            const availableThronglets = throngletsToPick.filter(t => 
                                t.id !== selectedThronglet.id && !t.isCurrentlySpeaking
                            );
                            
                            if (availableThronglets.length > 0) {
                                const thronglet = availableThronglets[Math.floor(Math.random() * availableThronglets.length)];
                                practicePhrase(phrasesToPractice[i].phrase, thronglet.id);
                            }
                        }, i * 5000); // 5 seconds between phrases
                    }
                    
                    // Some chance for Thronglets to repeat phrases they hear
                    setTimeout(() => {
                        if (Math.random() < 0.4 && window.thronglets.length > 1) {
                            // Pick a different Thronglet to repeat the first phrase
                            const otherThronglets = window.thronglets.filter(t => 
                                t.id !== selectedThronglet.id && !t.isCurrentlySpeaking
                            );
                            
                            if (otherThronglets.length > 0) {
                                const repeatingThronglet = otherThronglets[Math.floor(Math.random() * otherThronglets.length)];
                                practicePhrase(phrasesToPractice[0].phrase, repeatingThronglet.id);
                            }
                        }
                    }, 3000); // 3 second delay before repetition
                }
                
                return; // Exit the function after practicing phrases
            }
        }
        
        // Get word suggestions from LLM
        const suggestions = await getWordSuggestionsFromLLM();
        
        // Teach each suggested word that hasn't been taught before
        for (const suggestion of suggestions) {
            const word = suggestion.word.toLowerCase().trim();
            const associations = suggestion.associations || {};
            
            // Skip if this word is already in vocabulary
            if (vocabularyBase.has(word)) {
                console.log(`Skipping word "${word}" - already in vocabulary`);
                continue;
            }
            
            // Skip if this word was recently taught
            if (autoTaughtWords.some(item => item.word === word)) {
                console.log(`Skipping word "${word}" - recently taught`);
                continue;
            }
            
            // ENHANCEMENT: Ensure word has a value judgment (good or bad)
            if (!associations.good && !associations.bad) {
                // Add a value judgment based on the word's nature
                // This is a simplified heuristic - could be more sophisticated
                const positiveWords = ["happy", "fun", "love", "play", "friend", "help"];
                const negativeWords = ["sad", "angry", "hurt", "hungry", "sick", "pain"];
                
                let isPositive = positiveWords.some(pw => word.includes(pw));
                let isNegative = negativeWords.some(nw => word.includes(nw));
                
                // If no clear signal, use the associations to determine sentiment
                if (!isPositive && !isNegative) {
                    const assocWords = Object.keys(associations);
                    isPositive = assocWords.some(aw => positiveWords.includes(aw));
                    isNegative = assocWords.some(aw => negativeWords.includes(aw));
                }
                
                // Default to positive if still ambiguous
                if (isPositive || !isNegative) {
                    associations.good = 0.7;
                    console.log(`Added "good" association to "${word}"`);
                } else {
                    associations.bad = 0.7;
                    console.log(`Added "bad" association to "${word}"`);
                }
            }
            
            // ENHANCEMENT: Create more natural context-based learning rather than direct associations
            // Formulate simple sentences for the Thronglets to learn from
            let learningSentences = [];
            const existingVocab = Array.from(vocabularyBase.keys());
            
            // Get existing words to build contextual sentences
            const relevantExistingWords = existingVocab.filter(existingWord => 
                associations[existingWord] || 
                (existingWord.length > 2 && 
                 (word.includes(existingWord) || existingWord.includes(word)))
            );
            
            // Create contextual sentences with the new word and existing vocabulary
            if (relevantExistingWords.length > 0) {
                // Basic sentence with I/we + word
                if (vocabularyBase.has("I")) {
                    learningSentences.push(`I ${word}`);
                }
                
                if (vocabularyBase.has("we")) {
                    learningSentences.push(`we ${word}`);
                }
                
                // Sentences with associated words
                relevantExistingWords.forEach(existingWord => {
                    learningSentences.push(`${word} ${existingWord}`);
                    
                    // Sometimes create a more complex sentence
                    if (Math.random() < 0.5 && vocabularyBase.has("I")) {
                        learningSentences.push(`I ${word} ${existingWord}`);
                    }
                });
                
                // Add value judgment sentences
                if (associations.good) {
                    learningSentences.push(`${word} good`);
                } else if (associations.bad) {
                    learningSentences.push(`${word} bad`);
                }
            } else {
                // If no relevant words, create simple sentences
                learningSentences = [
                    `${word}`,
                    `I ${word}`,
                    `${word} good`,
                    `learn ${word}`
                ];
            }
            
            console.log(`Auto-teaching word: "${word}" with associations:`, associations);
            
            // Teach the word to all Thronglets
            const taught = teachWord(word, associations);
            
            if (taught) {
                // IMPORTANT FIX: Double-check that the word was added to vocabularyBase
                // This handles the case where teachWord might fail to add it
                if (!vocabularyBase.has(word)) {
                    // Generate Morse pattern
                    const morsePattern = textToMorse(word);
                    
                    // Add to vocabularyBase with auto-teaching specific metadata
                    vocabularyBase.set(word, {
                        word: word,
                        category: "auto_taught",
                        contexts: ["auto_teaching"],
                        weight: 0.5,
                        learnedFrom: "llm_suggestion",
                        learned: Date.now(),
                        lastUsed: Date.now(),
                        associations: [],
                        morsePattern: morsePattern,
                        autoTeachReason: suggestion.reason || "Auto-suggested by LLM"
                    });
                    
                    console.log(`Ensured "${word}" is added to vocabularyBase`);
                }
                
                // For each association, create bidirectional association
                for (const [assocWord, strength] of Object.entries(associations)) {
                    if (vocabularyBase.has(assocWord)) {
                        // Create reciprocal association
                        associateWords(assocWord, word, strength, "auto-teaching");
                        console.log(`Created reciprocal association: "${assocWord}" -> "${word}" (${strength})`);
                    }
                }
                
                // Add to auto-taught history
                autoTaughtWords.unshift({
                    word: word,
                    associations: {...associations},
                    timestamp: Date.now(),
                    reason: suggestion.reason || "Automatically taught"
                });
                
                // Keep history limited
                if (autoTaughtWords.length > MAX_AUTO_TAUGHT_HISTORY) {
                    autoTaughtWords.pop();
                }
                
                console.log(`Successfully taught word: ${word}`);
                
                // Notify a random Thronglet if available
                if (window.thronglets && window.thronglets.length > 0) {
                    const randomIndex = Math.floor(Math.random() * window.thronglets.length);
                    const randomThronglet = window.thronglets[randomIndex];
                    
                    // Create a string of associated concepts
                    let associationStr = Object.keys(associations).slice(0, 2).join(", ");
                    if (associationStr) {
                        associationStr = ` (${associationStr})`;
                    }
                    
                    randomThronglet.showThought(`I learned "${word}"!${associationStr}`);
                    
                    // ENHANCEMENT: Have the Thronglet activate the word in their brain
                    setTimeout(() => {
                        // Make sure this Thronglet vocalizes the word to strengthen neural connections
                        activateWordInBrain(word, randomThronglet.id);
                        
                        // Have other Thronglets practice the word too, but with a delay between each
                        if (window.thronglets.length > 1) {
                            // Randomly select 1-3 additional Thronglets to practice the word
                            const otherThronglets = window.thronglets.filter(t => t.id !== randomThronglet.id);
                            const maxPractice = Math.min(3, otherThronglets.length);
                            const practiceCount = Math.floor(Math.random() * maxPractice) + 1;
                            
                            // Shuffle and select Thronglets
                            const selectedThronglets = otherThronglets
                                .sort(() => Math.random() - 0.5)
                                .slice(0, practiceCount);
                            
                            // Have each selected Thronglet practice with increasing delays
                            selectedThronglets.forEach((thronglet, index) => {
                                setTimeout(() => {
                                    activateWordInBrain(word, thronglet.id);
                                }, (index + 1) * 3000); // 3 seconds between each Thronglet
                            });
                        }
                    }, 2000); // 2 second delay after initial learning
                    
                    // Visual indication of learning if the function exists
                    if (typeof window.createLearningSparkle === 'function' && randomThronglet.element) {
                        window.createLearningSparkle(randomThronglet.element, `Auto-Learned: ${word}`);
                    }
                }
                
                // Add to language memory for all Thronglets (if available)
                if (window.thronglets) {
                    window.thronglets.forEach(thronglet => {
                        if (thronglet && thronglet.id) {
                            // Add each learning sentence to the thronglet's experience
                            learningSentences.forEach(sentence => {
                                learnFromSentence(sentence, "auto_teaching", thronglet.id);
                            });
                        }
                    });
                }
                
                // We successfully taught one word, so we can stop for this cycle
                break;
            }
        }
    } catch (error) {
        console.error("Error in auto-teaching system:", error);
    }
}

/**
 * Save the current language system state to localStorage
 * @returns {boolean} Success status
 */
function saveLanguageSystemState() {
    try {
        // Create a state object with all important data
        const stateData = {
            timestamp: Date.now(),
            vocabulary: {},
            wordAssociations: wordAssociations,
            languageMemory: languageMemory,
            languageEnabled: languageEnabled,
            isAutoTeachEnabled: isAutoTeachEnabled,
            autoTaughtWords: autoTaughtWords
        };
        
        // Convert Map to a serializable object
        vocabularyBase.forEach((value, key) => {
            stateData.vocabulary[key] = value;
        });
        
        // Save to localStorage
        localStorage.setItem('throngletLanguageSystem', JSON.stringify(stateData));
        console.log("Language system state saved to localStorage");
        return true;
    } catch (error) {
        console.error("Error saving language system state:", error);
        return false;
    }
}

/**
 * Load language system state from localStorage
 * @returns {boolean} Success status
 */
function loadLanguageSystemState() {
    try {
        // Get saved data from localStorage
        const savedData = localStorage.getItem('throngletLanguageSystem');
        if (!savedData) {
            console.log("No saved language system state found");
            return false;
        }
        
        // Parse the data
        const stateData = JSON.parse(savedData);
        
        // Restore vocabulary
        vocabularyBase.clear();
        if (stateData.vocabulary) {
            for (const [key, value] of Object.entries(stateData.vocabulary)) {
                vocabularyBase.set(key, value);
            }
        }
        
        // Restore other data structures
        if (stateData.wordAssociations) {
            wordAssociations = stateData.wordAssociations;
        }
        
        if (stateData.languageMemory) {
            languageMemory = stateData.languageMemory;
        }
        
        // Restore auto-taught words history
        if (stateData.autoTaughtWords) {
            autoTaughtWords = stateData.autoTaughtWords;
            console.log(`Loaded ${autoTaughtWords.length} previously taught words`);
        }
        
        // Restore settings
        if (stateData.languageEnabled !== undefined) {
            languageEnabled = stateData.languageEnabled;
        }
        
        // Handle auto-teaching state
        if (stateData.isAutoTeachEnabled) {
            // Re-enable auto-teaching if it was enabled before
            setAutoTeachEnabled(true);
        }
        
        console.log("Language system state loaded from localStorage");
        console.log(`Loaded ${vocabularyBase.size} vocabulary items and ${languageMemory.length} memory entries`);
        return true;
    } catch (error) {
        console.error("Error loading language system state:", error);
        return false;
    }
}

/**
 * Save neural network weights for all thronglets
 * @returns {boolean} Success status
 */
function saveNeuralNetworkState() {
    try {
        if (!window.thronglets || window.thronglets.length === 0) {
            console.log("No thronglets available to save neural networks");
            return false;
        }
        
        const neuralNetworkData = {};
        
        // Save weights for each thronglet's neural network
        window.thronglets.forEach(thronglet => {
            if (thronglet && thronglet.id && thronglet.agent && thronglet.agent.weights) {
                neuralNetworkData[thronglet.id] = {
                    weights: thronglet.agent.weights,
                    timestamp: Date.now()
                };
                
                // Save concept weights if available
                if (thronglet.agent.weightsByConcept) {
                    neuralNetworkData[thronglet.id].weightsByConcept = thronglet.agent.weightsByConcept;
                }
                
                // Save concept knowledge if available
                if (thronglet.agent.conceptKnowledge) {
                    neuralNetworkData[thronglet.id].conceptKnowledge = thronglet.agent.conceptKnowledge;
                }
            }
        });
        
        // Save to localStorage
        localStorage.setItem('throngletNeuralNetworks', JSON.stringify(neuralNetworkData));
        console.log(`Neural network data saved for ${Object.keys(neuralNetworkData).length} thronglets`);
        return true;
    } catch (error) {
        console.error("Error saving neural network state:", error);
        return false;
    }
}

/**
 * Load neural network weights for all thronglets
 * @returns {boolean} Success status
 */
function loadNeuralNetworkState() {
    try {
        // Get saved data from localStorage
        const savedData = localStorage.getItem('throngletNeuralNetworks');
        if (!savedData || !window.thronglets || window.thronglets.length === 0) {
            console.log("No saved neural network data found or no thronglets available");
            return false;
        }
        
        // Parse the data
        const neuralNetworkData = JSON.parse(savedData);
        
        // Apply weights to matching thronglets
        let appliedCount = 0;
        window.thronglets.forEach(thronglet => {
            if (thronglet && thronglet.id && thronglet.agent && neuralNetworkData[thronglet.id]) {
                // Restore weights
                if (neuralNetworkData[thronglet.id].weights) {
                    thronglet.agent.weights = neuralNetworkData[thronglet.id].weights;
                }
                
                // Restore concept weights if available
                if (neuralNetworkData[thronglet.id].weightsByConcept) {
                    thronglet.agent.weightsByConcept = neuralNetworkData[thronglet.id].weightsByConcept;
                }
                
                // Restore concept knowledge if available
                if (neuralNetworkData[thronglet.id].conceptKnowledge) {
                    thronglet.agent.conceptKnowledge = neuralNetworkData[thronglet.id].conceptKnowledge;
                }
                
                appliedCount++;
            }
        });
        
        console.log(`Neural network data loaded for ${appliedCount} thronglets`);
        return appliedCount > 0;
    } catch (error) {
        console.error("Error loading neural network state:", error);
        return false;
    }
}

/**
 * Save the complete game state
 * @returns {boolean} Success status
 */
function saveGameState() {
    try {
        const languageSaved = saveLanguageSystemState();
        const networkSaved = saveNeuralNetworkState();
        
        return languageSaved || networkSaved;
    } catch (error) {
        console.error("Error saving game state:", error);
        return false;
    }
}

/**
 * Load the complete game state
 * @returns {boolean} Success status
 */
function loadGameState() {
    try {
        const languageLoaded = loadLanguageSystemState();
        const networkLoaded = loadNeuralNetworkState();
        
        return languageLoaded || networkLoaded;
    } catch (error) {
        console.error("Error loading game state:", error);
        return false;
    }
}

/**
 * Get the teaching history in a user-friendly format
 * @param {number} limit - Maximum number of items to return (default: 0 means all)
 * @returns {Array} - Array of teaching history items
 */
function getTeachingHistory(limit = 0) {
    if (autoTaughtWords.length === 0) {
        return [];
    }
    
    // Format the history items
    const historyItems = autoTaughtWords.map(item => {
        const timeAgo = Math.floor((Date.now() - item.timestamp) / 60000); // Minutes ago
        const timeString = timeAgo < 1 ? "just now" : 
                         timeAgo === 1 ? "1 minute ago" : 
                         `${timeAgo} minutes ago`;
        
        return {
            word: item.word,
            associations: item.associations,
            reason: item.reason || "Auto-taught",
            timeAgo: timeString,
            timestamp: item.timestamp
        };
    });
    
    // Limit the results if requested
    if (limit > 0 && limit < historyItems.length) {
        return historyItems.slice(0, limit);
    }
    
    return historyItems;
}

/**
 * Get example coherent phrases using the existing vocabulary, contextually relevant to a Thronglet's state
 * @param {Object} context - Context information about Thronglet states
 * @returns {Promise<Array>} - Array of example phrases
 */
async function getExamplePhrasesFromLLM(context = {}) {
    if (!modelLoaded) return [];
    
    try {
        // Prepare the context
        const vocabContext = getFormattedVocabularyForLLM();
        const throngletContext = getThrongletStatesInfo();
        
        // Build a context-specific prompt based on Thronglet states
        let contextualPrompt = "";
        if (context.states && context.states.length > 0) {
            contextualPrompt = "\nCONTEXTUAL REQUIREMENTS:\n";
            context.states.forEach(state => {
                contextualPrompt += `- Create phrases about being ${state.state} (${state.value}%)\n`;
            });
            contextualPrompt += "- Focus more on states with higher percentages\n";
        }
        
        // Include specific context for the selected Thronglet if available
        if (context.selectedThronglet) {
            contextualPrompt += `\nSPECIFIC THRONGLET CONTEXT:\n`;
            contextualPrompt += `- Thronglet #${context.selectedThronglet.id} has:\n`;
            contextualPrompt += `  * Hunger: ${context.selectedThronglet.hunger}%\n`;
            contextualPrompt += `  * Happiness: ${context.selectedThronglet.happiness}%\n`;
            
            // Determine primary needs
            if (context.selectedThronglet.hunger > 70) {
                contextualPrompt += `  * This Thronglet is VERY HUNGRY and needs food urgently\n`;
            } else if (context.selectedThronglet.hunger > 50) {
                contextualPrompt += `  * This Thronglet is hungry and thinking about food\n`;
            }
            
            if (context.selectedThronglet.happiness < 30) {
                contextualPrompt += `  * This Thronglet is SAD and needs to play\n`;
            } else if (context.selectedThronglet.happiness > 70) {
                contextualPrompt += `  * This Thronglet is VERY HAPPY and wants to share its joy\n`;
            }
        }
        
        const prompt = `
You are an AI assistant helping Thronglets (cute virtual creatures) learn language. 
Your task is to create example coherent phrases and sentences using their existing vocabulary.

${vocabContext}

${throngletContext}
${contextualPrompt}

INSTRUCTIONS:
1. Create 2-5 complete, coherent phrases or sentences using ONLY words that exist in the current vocabulary
2. Each phrase should be grammatically correct and demonstrate proper language use
3. Vary the complexity from simple (2-3 words) to more complex (4-6 words)
4. Include phrases that express the Thronglet's current state and needs based on the contextual requirements
5. Format your response as a JSON array with this structure:
[
  {
    "phrase": "I play with ball",
    "type": "action",
    "complexity": "simple",
    "context": "play"
  },
  {
    "phrase": "food makes me happy",
    "type": "emotion",
    "complexity": "medium",
    "context": "hunger"
  }
]

Please respond ONLY with the JSON array. Use ONLY words that appear in the vocabulary list above.`;

        // Truncate context if needed
        const truncatedPrompt = truncateContext(prompt);
        
        // Call Fireworks API
        const response = await fetch(FIREWORKS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${FIREWORKS_API_KEY}`
            },
            body: JSON.stringify({
                model: FIREWORKS_MODEL,
                messages: [
                    {
                        role: "user",
                        content: truncatedPrompt
                    }
                ],
                temperature: 0.7,
                max_tokens: 1000
            })
        });
        
        if (!response.ok) {
            throw new Error(`API response error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Extract the JSON response
        const content = data.choices[0].message.content;
        console.log("Raw LLM example phrases response:", content);
        
        // Parse the JSON response
        let examplePhrases = [];
        
        try {
            // Try to directly parse the entire content
            examplePhrases = JSON.parse(content);
        } catch (parseError) {
            console.log("Direct JSON parsing failed, trying extraction methods");
            
            try {
                // Try to extract JSON using regex (look for array pattern)
                const jsonMatch = content.match(/\[\s*\{[\s\S]*\}\s*\]/);
                if (jsonMatch) {
                    examplePhrases = JSON.parse(jsonMatch[0]);
                } else {
                    // Try to find individual objects and combine them
                    const objectMatches = content.match(/\{\s*"phrase"[\s\S]*?\}\s*(?=,|\]|$)/g);
                    if (objectMatches && objectMatches.length > 0) {
                        // Combine objects into a proper array
                        const jsonStr = '[' + objectMatches.join(',') + ']';
                        examplePhrases = JSON.parse(jsonStr);
                    } else {
                        // Last resort - try to manually extract the phrases
                        const phraseMatches = content.match(/"phrase"\s*:\s*"([^"]*)"/g);
                        if (phraseMatches && phraseMatches.length > 0) {
                            // Create basic objects with just the phrases
                            examplePhrases = phraseMatches.map(match => {
                                const phrase = match.match(/"phrase"\s*:\s*"([^"]*)"/)[1];
                                return { 
                                    phrase: phrase,
                                    type: "extracted",
                                    complexity: "unknown",
                                    context: detectPhraseContext(phrase)
                                };
                            });
                        }
                    }
                }
            } catch (extractError) {
                console.error("All JSON extraction methods failed:", extractError);
                
                // Use regex to find any quoted text as a last resort
                const quotedTextMatches = content.match(/"([^"]+)"/g);
                if (quotedTextMatches && quotedTextMatches.length > 0) {
                    // Filter to likely phrases (multiple words)
                    const phrases = quotedTextMatches
                        .map(match => match.replace(/"/g, ''))
                        .filter(text => text.split(/\s+/).length >= 2)
                        .filter(text => !text.includes(":") && !text.includes("{") && !text.includes("}"));
                    
                    examplePhrases = phrases.map(phrase => ({
                        phrase: phrase,
                        type: "extracted",
                        complexity: phrase.split(/\s+/).length <= 3 ? "simple" : "medium",
                        context: detectPhraseContext(phrase)
                    }));
                } else {
                    // Create contextual fallback examples if everything fails
                    examplePhrases = createContextualFallbackPhrases(context);
                }
            }
        }
        
        // Validate format
        if (!Array.isArray(examplePhrases)) {
            console.warn("Example phrases is not an array, converting");
            if (typeof examplePhrases === 'object' && examplePhrases.phrase) {
                // If a single object was returned, convert to array
                examplePhrases = [examplePhrases];
            } else {
                examplePhrases = [];
            }
        }
        
        // Filter out invalid phrases
        examplePhrases = examplePhrases.filter(item => 
            item && typeof item === 'object' && 
            item.phrase && typeof item.phrase === 'string');
        
        // Ensure each phrase has required fields and detect context if missing
        examplePhrases = examplePhrases.map(item => {
            if (!item.type) item.type = "general";
            if (!item.complexity) {
                const wordCount = item.phrase.split(/\s+/).length;
                item.complexity = wordCount <= 3 ? "simple" : "medium";
            }
            if (!item.context) {
                item.context = detectPhraseContext(item.phrase);
            }
            return item;
        });
        
        console.log("Processed example phrases:", examplePhrases);
        return examplePhrases;
    } catch (error) {
        console.error("Error getting example phrases from LLM:", error);
        // Return contextual fallback phrases
        return createContextualFallbackPhrases(context);
    }
}

/**
 * Detect the context of a phrase based on keywords
 * @param {string} phrase - The phrase to analyze
 * @returns {string} - Detected context (hunger, happiness, play, etc.)
 */
function detectPhraseContext(phrase) {
    phrase = phrase.toLowerCase();
    
    // Check for hunger-related keywords
    if (phrase.includes("food") || phrase.includes("eat") || 
        phrase.includes("hungry") || phrase.includes("apple")) {
        return "hunger";
    }
    
    // Check for happiness/play-related keywords
    if (phrase.includes("happy") || phrase.includes("play") || 
        phrase.includes("fun") || phrase.includes("ball") || 
        phrase.includes("joy")) {
        return "happiness";
    }
    
    // Check for social interaction
    if (phrase.includes("friend") || phrase.includes("together") || 
        phrase.includes("hello") || phrase.includes("we")) {
        return "social";
    }
    
    // Default to general
    return "general";
}

/**
 * Create contextual fallback phrases based on Thronglet state
 * @param {Object} context - Context information about Thronglet states
 * @returns {Array} - Array of basic phrases
 */
function createContextualFallbackPhrases(context) {
    const phrases = [];
    
    // Add some default phrases
    phrases.push({
        phrase: "I like food",
        type: "fallback",
        complexity: "simple",
        context: "hunger"
    });
    
    phrases.push({
        phrase: "playing makes me happy",
        type: "fallback", 
        complexity: "medium",
        context: "happiness"
    });
    
    // Add phrases based on context if available
    if (context.selectedThronglet) {
        // Hunger-related phrases
        if (context.selectedThronglet.hunger > 70) {
            phrases.push({
                phrase: "I am very hungry",
                type: "need",
                complexity: "simple",
                context: "hunger"
            });
            
            phrases.push({
                phrase: "need food now please",
                type: "request",
                complexity: "medium",
                context: "hunger"
            });
        } 
        else if (context.selectedThronglet.hunger > 50) {
            phrases.push({
                phrase: "I want food",
                type: "need",
                complexity: "simple",
                context: "hunger"
            });
        }
        
        // Happiness-related phrases
        if (context.selectedThronglet.happiness < 30) {
            phrases.push({
                phrase: "I am sad",
                type: "emotion",
                complexity: "simple",
                context: "happiness"
            });
            
            phrases.push({
                phrase: "want to play with ball",
                type: "request",
                complexity: "medium",
                context: "happiness"
            });
        } 
        else if (context.selectedThronglet.happiness > 70) {
            phrases.push({
                phrase: "I am happy",
                type: "emotion",
                complexity: "simple",
                context: "happiness"
            });
            
            phrases.push({
                phrase: "playing is fun",
                type: "observation",
                complexity: "simple",
                context: "happiness"
            });
        }
    }
    
    return phrases;
}

/**
 * Have a Thronglet practice saying a complete phrase
 * @param {string} phrase - The phrase to practice
 * @param {number} throngletId - The ID of the Thronglet practicing the phrase
 */
function practicePhrase(phrase, throngletId) {
    // Find the Thronglet with the given ID
    if (!window.thronglets) return;
    
    const thronglet = window.thronglets.find(t => t.id === parseInt(throngletId, 10));
    if (!thronglet) return;
    
    // Make sure the Thronglet isn't currently speaking
    if (thronglet.isCurrentlySpeaking) return;
    
    // Have the Thronglet say the phrase with a special thought bubble format
    thronglet.showThought(`💬 "${phrase}" 💬`);
    
    console.log(`Thronglet #${throngletId} practiced the phrase: "${phrase}"`);
    
    // Learn from this sentence
    learnFromSentence(phrase, "phrase_practice", throngletId);
    
    // If the Thronglet has an agent, reinforce this concept formation
    if (thronglet.agent) {
        // Get words from the phrase
        const words = phrase.toLowerCase()
            .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "")
            .split(/\s+/)
            .filter(w => w.length >= 2);
        
        // Initialize concept knowledge if it doesn't exist
        if (!thronglet.agent.conceptKnowledge) {
            thronglet.agent.conceptKnowledge = {};
        }
        
        // Strengthen connections between words in the phrase
        words.forEach(word => {
            // Strengthen the concept for each word
            if (!thronglet.agent.conceptKnowledge[word]) {
                thronglet.agent.conceptKnowledge[word] = 0.1;
            } else {
                thronglet.agent.conceptKnowledge[word] = 
                    Math.min(1.0, thronglet.agent.conceptKnowledge[word] + 0.03);
            }
            
            // Create associations between words in the phrase
            for (let i = 0; i < words.length; i++) {
                if (words[i] !== word) {
                    if (!thronglet.agent.weightsByConcept) {
                        thronglet.agent.weightsByConcept = {};
                    }
                    
                    if (!thronglet.agent.weightsByConcept[word]) {
                        thronglet.agent.weightsByConcept[word] = {};
                    }
                    
                    // Strengthen association between these words
                    thronglet.agent.weightsByConcept[word][words[i]] = 
                        (thronglet.agent.weightsByConcept[word][words[i]] || 0) + 0.02;
                }
            }
        });
    }
    
    // Record this practice in the Thronglet's memory
    if (window.addThrongletMemory) {
        window.addThrongletMemory(throngletId, 'phrase_practice', 
            `I practiced saying "${phrase}" to learn better language.`);
    }
}

// Export functions
window.ThrongletLanguage = {
    initialize: initializeLanguageModel,
    processInput: processLanguageInput,
    teachWord: teachWord,
    getStats: getVocabularyStats,
    getVocabulary: getFullVocabulary,
    reset: resetLanguageSystem,
    isEnabled: () => languageEnabled,
    setEnabled: (value) => { languageEnabled = value; },
    isLoaded: () => modelLoaded,
    recordEvent: recordLanguageEvent,
    processMorseCode: processMorseCode,
    getConceptAsMorse: getConceptAsMorse,
    textToMorse: textToMorse,
    morseToText: morseToText,
    extractMorsePatterns: extractMorsePatterns,
    // Auto-teaching functions
    setAutoTeachEnabled: setAutoTeachEnabled,
    isAutoTeachEnabled: () => isAutoTeachEnabled,
    getTeachingHistory: getTeachingHistory,
    // Save/load functions
    saveGameState: saveGameState,
    loadGameState: loadGameState,
    saveLanguageSystemState: saveLanguageSystemState,
    loadLanguageSystemState: loadLanguageSystemState,
    saveNeuralNetworkState: saveNeuralNetworkState,
    loadNeuralNetworkState: loadNeuralNetworkState
}; 