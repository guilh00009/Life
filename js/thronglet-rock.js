// Thronglet Rock System
// Adds rocks that can be placed and will fall, potentially killing Thronglets

// Global variables
let rocks = [];
let isPlacingRock = false;
const FEAR_RADIUS = 150; // How far Thronglets can see rocks
const FEAR_INCREASE_WITNESS = 0.3; // Fear increase when witnessing a death
const FEAR_INCREASE_NEARBY = 0.2; // Fear increase when near a falling rock
const FEAR_DECAY = 0.001; // How quickly fear decays over time

// Constants for neural network integration
const ROCK_FEAR_LEARNING_RATE = 0.2;
const ROCK_WITNESS_LEARNING_RATE = 0.3;
const ROCK_DANGER_RADIUS = 150;
const INITIAL_HARDCODED_ENCOUNTERS = 3; // Number of initial forced reactions

// Map to track rock encounters per Thronglet
let throngletRockEncounters = new Map();

// Forward declaration placeholder if needed, though hoisting should work
// function teachRockDangerToNearbyThronglets(rock) {}

// Function to integrate rock danger into a Thronglet's neural network
function integrateRockDangerToNeuralNetwork(thronglet, distance) {
    if (!thronglet.agent) return;
    
    // Add rock danger weights if they don't exist
    if (!thronglet.agent.weights.rock_danger) {
        thronglet.agent.weights.rock_danger = 0.1; // Initial weight
    }
    if (!thronglet.agent.weights.rock_avoidance) {
        thronglet.agent.weights.rock_avoidance = 0.1;
    }
    
    // Calculate learning strength based on distance
    const proximityFactor = 1 - (distance / ROCK_DANGER_RADIUS);
    const learningStrength = ROCK_FEAR_LEARNING_RATE * proximityFactor;
    
    // Update weights in neural network
    thronglet.agent.weights.rock_danger += learningStrength;
    thronglet.agent.weights.rock_avoidance += learningStrength;
    
    // Normalize weights to prevent extreme values
    thronglet.agent.weights.rock_danger = Math.min(1, thronglet.agent.weights.rock_danger);
    thronglet.agent.weights.rock_avoidance = Math.min(1, thronglet.agent.weights.rock_avoidance);
    
    // Add to concept knowledge if available
    if (thronglet.agent.conceptKnowledge) {
        if (!thronglet.agent.conceptKnowledge.rock_danger) {
            thronglet.agent.conceptKnowledge.rock_danger = 0.1;
        }
        thronglet.agent.conceptKnowledge.rock_danger += learningStrength;
        thronglet.agent.conceptKnowledge.rock_danger = 
            Math.min(1, thronglet.agent.conceptKnowledge.rock_danger);
    }
}

// Calculate fear response based on neural network weights
function calculateRockFearResponse(thronglet) {
    if (!thronglet.agent || !thronglet.agent.weights) return 0;
    
    // Combine different neural factors for fear response
    const dangerWeight = thronglet.agent.weights.rock_danger || 0;
    const avoidanceWeight = thronglet.agent.weights.rock_avoidance || 0;
    const conceptWeight = thronglet.agent.conceptKnowledge?.rock_danger || 0;
    
    // Calculate weighted average of fear factors
    return (dangerWeight + avoidanceWeight + conceptWeight) / 3;
}

// Function to warn nearby Thronglets (social learning)
function warnNearbyThronglets(warningThronglet, rock) {
    thronglets.forEach(otherThronglet => {
        if (otherThronglet.id === warningThronglet.id) return;
        
        const distance = Math.sqrt(
            Math.pow(warningThronglet.x - otherThronglet.x, 2) + 
            Math.pow(warningThronglet.y - otherThronglet.y, 2)
        );
        
        // If close enough to learn from warning
        if (distance < ROCK_DANGER_RADIUS / 2) {
            // Social learning - learn from other Thronglet's warning
            if (otherThronglet.agent) {
                const socialLearningStrength = ROCK_FEAR_LEARNING_RATE * 0.5; // Reduced learning from warnings
                integrateRockDangerToNeuralNetwork(otherThronglet, distance);
                
                // Add to Thronglet's memory
                if (typeof addThrongletMemory === 'function') {
                    addThrongletMemory(otherThronglet.id, 'warning_received', 
                        `Learned about rock danger from Thronglet #${warningThronglet.id}'s warning!`);
                }
            }
        }
    });
}

// Function to teach Thronglets about rock danger, incorporating initial hardcoded reactions
function teachRockDangerToNearbyThronglets(rock) {
    if (!thronglets) return;
    
    thronglets.forEach(thronglet => {
        if (!thronglet.agent) return;
        
        const distance = Math.sqrt(
            Math.pow(rock.x + 25 - thronglet.x, 2) + 
            Math.pow(rock.y + 25 - thronglet.y, 2)
        );
        
        // If Thronglet is within danger radius
        if (distance < ROCK_DANGER_RADIUS) {
            // Get current encounter count, default to 0
            let encounterCount = throngletRockEncounters.get(thronglet.id) || 0;
            
            // --- Learning Always Happens ---
            integrateRockDangerToNeuralNetwork(thronglet, distance);
            
            // --- Reaction Logic ---
            if (encounterCount < INITIAL_HARDCODED_ENCOUNTERS) {
                // **Hardcoded Initial Reaction**
                console.log(`Thronglet ${thronglet.id}: Hardcoded rock reaction #${encounterCount + 1}`);
                thronglet.showThought("ROCK! RUN! (Instinct) 😨");
                runFromRock(thronglet, rock);
                warnNearbyThronglets(thronglet, rock); // Warn others during hardcoded phase
                
                // Increment and store the encounter count
                throngletRockEncounters.set(thronglet.id, encounterCount + 1);
                
            } else {
                // **Learned Reaction (Neural Network Based)**
                const fearResponse = calculateRockFearResponse(thronglet);
                 console.log(`Thronglet ${thronglet.id}: Learned rock reaction (Fear: ${fearResponse.toFixed(2)})`);
                
                if (fearResponse > 0.7) {
                    thronglet.showThought("DANGER! ROCK! (Learned) 😱");
                    runFromRock(thronglet, rock);
                    warnNearbyThronglets(thronglet, rock); // Warn others if learned fear is high
                } else if (fearResponse > 0.3) {
                    thronglet.showThought("Careful! Rock! (Learned) 😨");
                    runFromRock(thronglet, rock);
                } else {
                    thronglet.showThought("A rock? (Learned) 🤔");
                    // Still learns slightly even if reaction is mild
                    thronglet.agent.learnFromObservation?.('rock_nearby', 0.05);
                }
            }
        }
    });
}

// Initialize rock system
function initializeRockSystem() {
    console.log("Initializing rock system...");
    
    // Add rock button to item buttons
    const itemButtons = document.getElementById('item-buttons');
    if (itemButtons) {
        const rockButton = document.createElement('button');
        rockButton.className = 'item-button';
        rockButton.id = 'rock-button';
        rockButton.textContent = '🪨 Place Rock';
        rockButton.addEventListener('click', toggleRockPlacement);
        itemButtons.appendChild(rockButton);
    }
    
    // Add rock placement event listener to game scene
    const gameScene = document.getElementById('game-scene');
    if (gameScene) {
        gameScene.addEventListener('click', handleRockPlacement);
    }
}

// Toggle rock placement mode
function toggleRockPlacement() {
    isPlacingRock = !isPlacingRock;
    
    // Update button appearance
    const rockButton = document.getElementById('rock-button');
    if (rockButton) {
        if (isPlacingRock) {
            rockButton.style.backgroundColor = '#e74c3c';
            rockButton.textContent = '🪨 Cancel Rock';
        } else {
            rockButton.style.backgroundColor = '#3498db';
            rockButton.textContent = '🪨 Place Rock';
        }
    }
    
    // Update cursor
    const gameScene = document.getElementById('game-scene');
    if (gameScene) {
        gameScene.style.cursor = isPlacingRock ? 'crosshair' : 'default';
    }
}

// Handle rock placement
function handleRockPlacement(event) {
    if (!isPlacingRock) return;
    
    const gameScene = document.getElementById('game-scene');
    if (!gameScene) return;
    
    // Get coordinates relative to game scene
    const rect = gameScene.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Create rock at clicked position
    createRock(x, y);
    
    // Exit placement mode
    toggleRockPlacement();
}

// Create a rock
function createRock(x, y) {
    const gameScene = document.getElementById('game-scene');
    if (!gameScene) return;
    
    const rockElement = document.createElement('div');
    rockElement.className = 'rock falling';
    
    const randomScale = 0.8 + Math.random() * 0.4;
    const randomRotation = -15 + Math.random() * 30;
    
    rockElement.style.transform = `rotate(${randomRotation}deg) scale(${randomScale})`;
    rockElement.style.left = x - 25 + 'px';
    rockElement.style.top = '0px';
    
    gameScene.appendChild(rockElement);
    
    const rock = {
        element: rockElement,
        x: x - 25,
        y: 0,
        targetY: y - 25,
        isFalling: true,
        hasKilled: false,
        scale: randomScale
    };
    
    // Teach nearby Thronglets about rock danger through their neural networks
    teachRockDangerToNearbyThronglets(rock);
    
    rocks.push(rock);
    animateRockFall();
}

// Animate rock falling
function animateRockFall() {
    // Fixed animation logic - previous check was inverted
    const isFallingAnimationRunning = rocks.some(rock => rock.isFalling);
    
    console.log("animateRockFall called, rocks falling:", isFallingAnimationRunning);
    console.log("Number of rocks:", rocks.length);
    
    // Remove this check that was preventing animation from starting
    // if (isFallingAnimationRunning) {
    //     // Animation already running
    //     return;
    // }
    
    // Start animation loop
    requestAnimationFrame(updateRockPositions);
}

// Update rock positions during fall
function updateRockPositions() {
    let isFallingAnimationRunning = false;
    
    for (let i = 0; i < rocks.length; i++) {
        const rock = rocks[i];
        
        if (rock.isFalling) {
            console.log(`Rock ${i} falling, current y: ${rock.y}, target: ${rock.targetY}`);
            
            // Apply gravity
            rock.y += 5; // Fall speed
            
            // Update position
            rock.element.style.top = rock.y + 'px';
            
            // Check if reached target
            if (rock.y >= rock.targetY) {
                rock.y = rock.targetY;
                rock.element.style.top = rock.y + 'px';
                rock.isFalling = false;
                
                console.log(`Rock ${i} landed at y: ${rock.y}`);
                
                // Check for collision with Thronglets during landing
                checkRockCollisions(rock);
                
                // Add impact effect
                createImpactEffect(rock.x + 25, rock.y + 25);
            } else {
                isFallingAnimationRunning = true;
            }
        }
    }
    
    // Continue animation if rocks are still falling
    if (isFallingAnimationRunning) {
        requestAnimationFrame(updateRockPositions);
    }
}

// Check for collisions between a falling rock and Thronglets
function checkRockCollisions(rock) {
    // Only check collisions right when the rock lands
    if (rock.isFalling || rock.hasKilled) return;
    
    // Get reference to thronglets array (from global scope)
    if (typeof thronglets === 'undefined') {
        console.error("Thronglets array not found");
        return;
    }
    
    console.log(`Checking collision for rock at y: ${rock.y}`);
    
    for (let i = 0; i < thronglets.length; i++) {
        const thronglet = thronglets[i];
        
        // Calculate distance between rock center and thronglet center
        const rockCenterX = rock.x + 25;
        const rockCenterY = rock.y + 25;
        const throngletCenterX = thronglet.x;
        const throngletCenterY = thronglet.y;
        
        const distance = Math.sqrt(
            Math.pow(rockCenterX - throngletCenterX, 2) + 
            Math.pow(rockCenterY - throngletCenterY, 2)
        );
        
        // Adjust collision threshold based on rock scale
        const collisionThreshold = 30 * rock.scale; // Slightly reduced threshold for better accuracy
        
        console.log(`Distance to Thronglet ${thronglet.id}: ${distance.toFixed(1)}, Threshold: ${collisionThreshold.toFixed(1)}`);
        
        // If thronglet is within collision range
        if (distance < collisionThreshold) {
            console.log(`Collision detected with Thronglet ${thronglet.id}!`);
            // Rock has landed on a thronglet
            killThronglet(thronglet, i); // Pass index `i`
            rock.hasKilled = true;
            break; // Stop checking after the first kill
        }
    }
}

// Kill a Thronglet that was hit by a falling rock
function killThronglet(thronglet, index) {
    console.log(`Thronglet #${thronglet.id} was killed by a falling rock!`);
    
    // Show death message
    thronglet.showThought("CRUSHED! 💀");
    
    // Trigger screen shake
    triggerScreenShake(5);
    
    // Create blood splatter effect
    createBloodSplatterEffect(thronglet.x, thronglet.y);
    
    // Make nearby Thronglets learn from this death through their neural networks
    thronglets.forEach(otherThronglet => {
        if (otherThronglet.id === thronglet.id) return;
        
        const distance = Math.sqrt(
            Math.pow(thronglet.x - otherThronglet.x, 2) + 
            Math.pow(thronglet.y - otherThronglet.y, 2)
        );
        
        // If close enough to witness the death
        if (distance < ROCK_DANGER_RADIUS) {
            // Strong learning from witnessing death
            if (otherThronglet.agent) {
                // Stronger learning rate for witnessing death
                const learningStrength = ROCK_WITNESS_LEARNING_RATE * (1 - distance / ROCK_DANGER_RADIUS);
                
                // Update neural network with death observation
                otherThronglet.agent.weights.rock_danger = 
                    Math.min(1, (otherThronglet.agent.weights.rock_danger || 0) + learningStrength);
                otherThronglet.agent.weights.rock_avoidance = 
                    Math.min(1, (otherThronglet.agent.weights.rock_avoidance || 0) + learningStrength);
                
                // Update concept knowledge
                if (otherThronglet.agent.conceptKnowledge) {
                    otherThronglet.agent.conceptKnowledge.rock_danger = 
                        Math.min(1, (otherThronglet.agent.conceptKnowledge.rock_danger || 0) + learningStrength);
                }
                
                // React based on learning
                otherThronglet.showThought("NO! They got crushed! Must avoid rocks! 😱");
                runFromRock(otherThronglet, { x: thronglet.x - 25, y: thronglet.y - 25 });
                
                // Add to memory
                if (typeof addThrongletMemory === 'function') {
                    addThrongletMemory(otherThronglet.id, 'witness_death', 
                        `I saw Thronglet #${thronglet.id} get crushed! Rocks are very dangerous!`);
                }
            }
        }
    });
    
    // Rest of killThronglet function...
    thronglet.isMoving = false;
    thronglet.element.style.display = 'none';
    thronglet.statsElement.style.display = 'none';
    
    let removed = false;
    if (index >= 0 && index < thronglets.length && thronglets[index] === thronglet) {
        console.log(`Removing Thronglet at index ${index}`);
        thronglets.splice(index, 1);
        removed = true;
    } else {
        console.error("Error removing Thronglet: Index mismatch or Thronglet not found");
        const actualIndex = thronglets.findIndex(t => t.id === thronglet.id);
        if (actualIndex !== -1) {
            console.log(`Fallback: Removing Thronglet at index ${actualIndex}`);
            thronglets.splice(actualIndex, 1);
            removed = true;
        }
    }

    if (removed) {
        updateThrongletCounter();
    }

    setTimeout(() => {
        if (thronglet.element && thronglet.element.parentNode) {
            thronglet.element.parentNode.removeChild(thronglet.element);
        }
        if (thronglet.statsElement && thronglet.statsElement.parentNode) {
            thronglet.statsElement.parentNode.removeChild(thronglet.statsElement);
        }
    }, 100);
}

// Function to update the Thronglet counter display
function updateThrongletCounter() {
    const counterElement = document.getElementById('thronglet-counter');
    if (counterElement) {
        counterElement.textContent = `Thronglets: ${thronglets.length}`;
        console.log("Updated Thronglet counter to:", thronglets.length);
    } else {
        console.warn("Thronglet counter element not found");
    }
}

// Create impact effect when rock hits the ground
function createImpactEffect(x, y) {
    const gameScene = document.getElementById('game-scene');
    if (!gameScene) return;
    
    // Create impact element
    const impactElement = document.createElement('div');
    impactElement.className = 'impact-effect';
    impactElement.style.left = (x - 25) + 'px';
    impactElement.style.top = (y - 25) + 'px';
    
    // Add to game scene
    gameScene.appendChild(impactElement);
    
    // Remove after animation completes
    setTimeout(() => {
        if (impactElement.parentNode) {
            impactElement.parentNode.removeChild(impactElement);
        }
    }, 1000);
}

// Create death effect when a Thronglet is killed
function createDeathEffect(x, y) {
    const gameScene = document.getElementById('game-scene');
    if (!gameScene) return;
    
    // Create death effect element
    const deathElement = document.createElement('div');
    deathElement.className = 'death-effect';
    deathElement.style.left = (x - 25) + 'px';
    deathElement.style.top = (y - 25) + 'px';
    
    // Add to game scene
    gameScene.appendChild(deathElement);
    
    // Remove after animation completes
    setTimeout(() => {
        if (deathElement.parentNode) {
            deathElement.parentNode.removeChild(deathElement);
        }
    }, 2000);
}

// Create blood splatter effect
function createBloodSplatterEffect(x, y) {
    const gameScene = document.getElementById('game-scene');
    if (!gameScene) return;
    
    const particleCount = 15;
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'blood-particle';
        
        // Position near the impact point
        particle.style.left = (x - 5 + Math.random() * 10) + 'px';
        particle.style.top = (y - 5 + Math.random() * 10) + 'px';
        
        // Randomize animation properties
        const angle = Math.random() * Math.PI * 2;
        const distance = 50 + Math.random() * 50;
        const duration = 0.5 + Math.random() * 0.5;
        
        particle.style.setProperty('--tx', `${Math.cos(angle) * distance}px`);
        particle.style.setProperty('--ty', `${Math.sin(angle) * distance}px`);
        particle.style.animationDuration = `${duration}s`;
        
        gameScene.appendChild(particle);
        
        // Remove particle after animation
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
            }
        }, duration * 1000);
    }
}

// Trigger screen shake effect
function triggerScreenShake(intensity = 5) {
    const gameContainer = document.getElementById('game-container');
    if (!gameContainer) return;
    
    gameContainer.style.setProperty('--shake-intensity', `${intensity}px`);
    gameContainer.classList.add('shake');
    
    // Remove the class after the animation finishes
    setTimeout(() => {
        gameContainer.classList.remove('shake');
    }, 500); // Match animation duration
}

// Add a function to detect collision between a Thronglet and stationary rocks
function handleThrongletRockCollision() {
    // Run collision detection every frame
    requestAnimationFrame(handleThrongletRockCollision);
    
    // If no rocks or thronglets exist, just return
    if (typeof rocks === 'undefined' || rocks.length === 0 || 
        typeof thronglets === 'undefined' || thronglets.length === 0) {
        return;
    }
    
    // Check each thronglet against each rock
    for (let i = 0; i < thronglets.length; i++) {
        const thronglet = thronglets[i];
        
        // Skip if the thronglet isn't moving
        if (!thronglet.isMoving) continue;
        
        // Check against all rocks that are not falling
        for (let j = 0; j < rocks.length; j++) {
            const rock = rocks[j];
            
            // Skip falling rocks - they're handled separately
            if (rock.isFalling) continue;
            
            // Calculate distance between centers
            const rockCenterX = rock.x + 25;
            const rockCenterY = rock.y + 25;
            const throngletCenterX = thronglet.x;
            const throngletCenterY = thronglet.y;
            
            const distance = Math.sqrt(
                Math.pow(rockCenterX - throngletCenterX, 2) + 
                Math.pow(rockCenterY - throngletCenterY, 2)
            );
            
            // Define collision threshold based on rock size
            const collisionThreshold = 40 * rock.scale;
            
            // If thronglet is colliding with a stationary rock
            if (distance < collisionThreshold) {
                // Make the thronglet stop and show a reaction
                thronglet.targetX = null;
                thronglet.targetY = null;
                thronglet.isMoving = false;
                thronglet.setClassName('idle');
                
                // Show a reaction thought bubble
                thronglet.showThought("Ouch! A rock!");
                
                // Bounce the thronglet slightly away from the rock
                const dx = throngletCenterX - rockCenterX;
                const dy = throngletCenterY - rockCenterY;
                const bounceDistance = 20;
                
                // Normalize and apply bounce
                const length = Math.sqrt(dx * dx + dy * dy);
                if (length > 0) {
                    const newX = thronglet.x + (dx / length) * bounceDistance;
                    const newY = thronglet.y + (dy / length) * bounceDistance;
                    
                    // Apply new position with bounds checking
                    const gameScene = document.getElementById('game-scene');
                    if (gameScene) {
                        const width = gameScene.clientWidth;
                        const height = gameScene.clientHeight;
                        
                        thronglet.x = Math.max(20, Math.min(width - 20, newX));
                        thronglet.y = Math.max(20, Math.min(height - 20, newY));
                        
                        thronglet.element.style.left = thronglet.x + 'px';
                        thronglet.element.style.top = thronglet.y + 'px';
                        
                        // Update stats position
                        thronglet.updateStatsPosition();
                    }
                }
            }
        }
    }
}

// Function to warn Thronglets about falling rocks
function warnThrongletsAboutRock(rock) {
    if (!thronglets) return;
    
    thronglets.forEach(thronglet => {
        const distance = Math.sqrt(
            Math.pow(rock.x + 25 - thronglet.x, 2) + 
            Math.pow(rock.y + 25 - thronglet.y, 2)
        );
        
        // If Thronglet is within fear radius
        if (distance < FEAR_RADIUS) {
            // Get current fear level
            let fearLevel = throngletFearMemory.get(thronglet.id) || 0;
            
            // Increase fear
            fearLevel = Math.min(1, fearLevel + FEAR_INCREASE_NEARBY);
            throngletFearMemory.set(thronglet.id, fearLevel);
            
            // Make Thronglet react based on fear level
            if (fearLevel > 0.7) {
                // High fear - run away and warn others
                thronglet.showThought("ROCK! RUN! 😱");
                runFromRock(thronglet, rock);
            } else if (fearLevel > 0.3) {
                // Medium fear - just run
                thronglet.showThought("Careful! Rock! 😨");
                runFromRock(thronglet, rock);
            } else {
                // Low fear - just watch
                thronglet.showThought("A rock? 🤔");
            }
        }
    });
}

// Function to make Thronglet run from rock
function runFromRock(thronglet, rock) {
    const dx = thronglet.x - (rock.x + 25);
    const dy = thronglet.y - (rock.y + 25);
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance === 0) return;
    
    const escapeDistance = 100;
    const escapeX = thronglet.x + (dx / distance) * escapeDistance;
    const escapeY = thronglet.y + (dy / distance) * escapeDistance;
    
    const gameScene = document.getElementById('game-scene');
    if (gameScene) {
        const boundaryPadding = 50;
        const maxX = gameScene.clientWidth - boundaryPadding;
        const maxY = gameScene.clientHeight - boundaryPadding;
        
        const safeX = Math.max(boundaryPadding, Math.min(maxX, escapeX));
        const safeY = Math.max(boundaryPadding, Math.min(maxY, escapeY));
        
        thronglet.moveTowards(safeX, safeY);
    }
}

// Start fear update loop when window loads
window.addEventListener('load', () => {
    setTimeout(() => {
        initializeRockSystem();
        handleThrongletRockCollision();
        updateThrongletCounter();
    }, 1000);
}); 