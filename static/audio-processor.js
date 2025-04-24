class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 16000 * 2; // 2秒分のバッファ
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        this.lastProcessFrame = 0;
        this.processIntervalFrames = 2 * sampleRate; // 2秒分のフレーム数
    }

    process(inputs, outputs, parameters, currentFrame) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        // 入力データをバッファに追加
        const inputData = input[0];
        for (let i = 0; i < inputData.length; i++) {
            if (this.bufferIndex < this.bufferSize) {
                this.buffer[this.bufferIndex++] = inputData[i];
            }
        }

        // バッファが満杯になった場合に処理を実行
        if (this.bufferIndex >= this.bufferSize) {
            // バッファの内容を送信
            const audioData = this.buffer.slice(0, this.bufferIndex);
            this.port.postMessage(audioData.buffer, [audioData.buffer]);

            // バッファをリセット
            this.buffer = new Float32Array(this.bufferSize);
            this.bufferIndex = 0;
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
