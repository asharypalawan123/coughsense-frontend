import { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  audioLevel: number;
  isRecording: boolean;
}

export const WaveformVisualizer = ({ audioLevel, isRecording }: WaveformVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw waveform bars
    const barCount = 40;
    const barWidth = width / barCount;
    const centerY = height / 2;
    
    for (let i = 0; i < barCount; i++) {
      const barHeight = isRecording 
        ? (Math.sin(Date.now() / 200 + i / 2) * 0.5 + 0.5) * audioLevel * height * 0.8
        : 2;
      
      const x = i * barWidth;
      const gradient = ctx.createLinearGradient(0, centerY - barHeight/2, 0, centerY + barHeight/2);
      gradient.addColorStop(0, 'hsl(191, 91%, 36%)');
      gradient.addColorStop(1, 'hsl(191, 81%, 56%)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(x + 2, centerY - barHeight/2, barWidth - 4, barHeight);
    }
  }, [audioLevel, isRecording]);

  return (
    <div className="w-full max-w-md mx-auto bg-secondary/30 rounded-xl p-4 border border-primary/20">
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={100}
        className="w-full h-24"
      />
    </div>
  );
};
