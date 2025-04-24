class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input && input.length > 0) {
            const channel = input[0];
            
            for (let i = 0; i < channel.length; i++) {
                if (this.bufferIndex < this.bufferSize) {
                    this.buffer[this.bufferIndex++] = channel[i];
                }
            }
            
            if (this.bufferIndex >= this.bufferSize) {
                this.port.postMessage(this.buffer.buffer);
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
