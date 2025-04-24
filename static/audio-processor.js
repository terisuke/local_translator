class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 8192;  // バッファサイズ（0.5秒分のデータ）
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const channel = input[0];

        if (!channel) return true;

        // バッファにデータを追加
        for (let i = 0; i < channel.length; i++) {
            this.buffer[this.bufferIndex] = channel[i];
            this.bufferIndex++;

            // バッファが一杯になったら送信
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
