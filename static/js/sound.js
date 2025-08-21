// Sound notification functionality
function playSuccessSound() {
    // Create an oscillator for a success sound
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Create oscillator for a pleasant "ding" sound
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(1046.50, audioContext.currentTime); // C6
        oscillator.frequency.setValueAtTime(1318.51, audioContext.currentTime + 0.1); // E6
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.5);
        
        console.log('Success sound played');
    } catch (error) {
        console.error('Error playing sound:', error);
    }
}

// Export the function for use in other scripts
window.playSuccessSound = playSuccessSound;