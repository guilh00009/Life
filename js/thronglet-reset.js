/**
 * Thronglet Reset System
 * Provides functionality for resetting the game state
 */

/**
 * Resets the entire game state to factory defaults
 */
function resetEverything() {
    console.log("Resetting everything to factory defaults");
    
    // Clear all game entities
    thronglets.forEach(t => {
        if (t.element && t.element.parentNode) {
            t.element.parentNode.removeChild(t.element);
        }
    });
    
    apples.forEach(a => {
        if (a.element && a.element.parentNode) {
            a.element.parentNode.removeChild(a.element);
        }
    });
    
    balls.forEach(b => {
        if (b.element && b.element.parentNode) {
            b.element.parentNode.removeChild(b.element);
        }
    });
    
    appleTrees.forEach(t => {
        if (t.element && t.element.parentNode) {
            t.element.parentNode.removeChild(t.element);
        }
    });
    
    // Reset arrays
    thronglets = [];
    eggs = [];
    apples = [];
    balls = [];
    trees = [];
    appleTrees = [];
    
    // Reset all localStorage data
    localStorage.removeItem('throngletLanguageSystem');
    localStorage.removeItem('throngletNeuralNetworks');
    localStorage.removeItem('globalContext');
    localStorage.removeItem('throngletEvolution');
    localStorage.removeItem('featureVotes');
    localStorage.removeItem('conversationHistory');
    
    // Reset global variables
    nextThrongletId = 0;
    globalContext = {
        gameStartTime: Date.now(),
        sessionEvents: [],
        throngletMemory: {},
        appleHistory: [],
        ballHistory: [],
        creatorInteractions: []
    };
    
    // Reset feature tracking
    suggestedFeatures = [];
    featureVotes = {};
    conversationHistory = [];
    
    // Reset counters and visuals
    updateThrongletCounter();
    message.textContent = "Game reset! Click the egg 3 times to hatch a Thronglet!";
    
    // Restart egg
    if (egg && egg.style) {
        egg.style.display = 'block';
    }
    
    // Reset click count
    clicks = 0;
    
    // Update UI/state
    if (throngletCounter) {
        throngletCounter.textContent = "0";
    }
    
    // Show confirmation
    alert("Game completely reset! Start fresh with a new egg.");
    
    // Force page reload to ensure clean state
    window.location.reload();
} 